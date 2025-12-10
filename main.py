import os
import json
from collections import defaultdict, Counter
from typing import Optional

import pandas as pd
from openai import OpenAI

PROMPT_FILE = "prompt.txt"
OPENAI_MODEL = "gpt-4.1"
# API key is read from environment; do not hardcode in code.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

INPUT_DIRECTORY = "rm_notes"
NOTE_COLUMNS = ["note_content"]
NOTE_DATE_COLUMN = "note_created_at"
# NPS_SCORE_COLUMN = "nps_rating"
OUTPUT_DIRECTORY = "outputs"
OUTPUT_CSV_BASENAME = "nps_analysis_summary"
REQUIRED_COLUMNS = ["user_id", NOTE_DATE_COLUMN, *NOTE_COLUMNS]
NO_CONVERSATION_CATEGORY = "No Conversation Recorded"
NO_CONVERSATION_SUMMARY = (
    "Learner could not be reached (Did Not Pick). No conversation feedback recorded."
)


def load_system_prompt(prompt_path: str = PROMPT_FILE) -> str:
    """Load the system prompt text from disk."""
    try:
        with open(prompt_path, "r", encoding="utf-8") as prompt_file:
            return prompt_file.read().strip()
    except FileNotFoundError as exc:
        raise RuntimeError(f"System prompt file not found: {prompt_path}") from exc
    except OSError as exc:
        raise RuntimeError(f"Unable to read prompt file: {prompt_path}") from exc


def configure_openai_client() -> OpenAI:
    """Configure and return an OpenAI client."""
    api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before running this script.")
    return OpenAI(api_key=api_key)


SYSTEM_PROMPT = load_system_prompt()
# Lazy client initialization to allow UI/streamlit flows to set env before use.
openai_client: Optional[OpenAI] = None


def load_csv(file_path):
    """Load and validate CSV file"""
    try:
        df = pd.read_csv(file_path)
        print(f"✓ CSV loaded successfully: {len(df)} rows")
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            raise RuntimeError(
                f"Missing required columns in CSV: {', '.join(missing_columns)}"
            )
        return df
    except FileNotFoundError:
        print(f"✗ File not found: {file_path}")
        return None
    except Exception as e:
        print(f"✗ Error loading CSV: {e}")
        return None


def ensure_output_directory(path: str = OUTPUT_DIRECTORY) -> None:
    """Ensure that the output directory exists."""
    os.makedirs(path, exist_ok=True)


def resolve_input_csv(input_dir: str = INPUT_DIRECTORY) -> str:
    """Resolve the CSV input file inside the rm_notes directory."""
    if not os.path.isdir(input_dir):
        raise RuntimeError(f"Input directory not found: {input_dir}")

    csv_files = sorted(
        f for f in os.listdir(input_dir) if f.lower().endswith(".csv")
    )
    if not csv_files:
        raise RuntimeError(
            f"No CSV files found in input directory: {input_dir}"
        )

    return os.path.join(input_dir, csv_files[0])


def group_notes_by_user(df):
    """Group note content (with timestamps) by user_id using fixed column definitions."""
    grouped = defaultdict(lambda: {"notes": []})

    for idx, row in df.iterrows():
        user_id = row.get("user_id")

        if pd.isna(user_id):
            print(f"Warning: Row {idx} has missing user_id, skipping...")
            continue

        note_timestamp = row.get(NOTE_DATE_COLUMN)
        parsed_timestamp = None
        if pd.notna(note_timestamp):
            parsed_timestamp = pd.to_datetime(note_timestamp, errors="coerce")

        for note_col in NOTE_COLUMNS:
            note_value = row.get(note_col)
            if pd.notna(note_value):
                grouped[user_id]["notes"].append({
                    "content": str(note_value),
                    "date": parsed_timestamp,
                })

    return grouped


def filter_by_nps(grouped_data, target_scores=None):
    """Return all users (NPS filtering removed)"""
    return grouped_data


def consolidate_user_notes(notes):
    """Consolidate multiple notes into a single text, ordered chronologically."""
    if not notes:
        return "No notes available"

    def sort_key(note):
        note_date = note.get("date")
        if note_date is None or (isinstance(note_date, pd.Timestamp) and pd.isna(note_date)):
            return (pd.Timestamp.max, "")
        return (note_date, note.get("content", ""))

    ordered_notes = sorted(notes, key=sort_key)
    formatted_entries = []
    for note in ordered_notes:
        content = str(note.get("content", "")).strip()
        note_date = note.get("date")
        if isinstance(note_date, pd.Timestamp) and not pd.isna(note_date):
            date_str = note_date.strftime("%Y-%m-%d %H:%M")
        elif note_date:
            date_str = str(note_date)
        else:
            date_str = "Unknown Date"
        formatted_entries.append(f"[{date_str}] {content}")
    return " | ".join(formatted_entries)


def notes_are_dnp_only(notes) -> bool:
    """Return True if every note represents a Did Not Pick (DNP) status."""
    cleaned = []
    for note in notes:
        content = str(note.get("content", "")).strip()
        if not content:
            continue
        normalized = content.upper()
        cleaned.append(normalized)

    if not cleaned:
        return False

    def is_dnp_entry(text: str) -> bool:
        stripped = text.lstrip()
        return stripped.startswith("DNP")

    return all(is_dnp_entry(entry) for entry in cleaned)


def get_openai_client() -> OpenAI:
    """Lazy-init OpenAI client so env can be set before first call."""
    global openai_client
    if openai_client is None:
        openai_client = configure_openai_client()
    return openai_client


def analyze_user_feedback(user_id, notes):
    """Analyze feedback for a single user using ChatGPT (GPT-4.1)."""
    consolidated_notes = consolidate_user_notes(notes)

    if not notes or consolidated_notes == "No notes available":
        return {
            "user_id": user_id,
            "error": "Insufficient data for analysis"
        }

    if notes_are_dnp_only(notes):
        return {
            "user_id": user_id,
            "categories": [NO_CONVERSATION_CATEGORY],
            "consolidated_summary": NO_CONVERSATION_SUMMARY,
        }

    prompt = f"""Analyze the following customer feedback:

{consolidated_notes}

Provide analysis in the required JSON format."""

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )

        response_text = (response.choices[0].message.content or "").strip()
        analysis = json.loads(response_text)

        # Validate that categories exist and are not empty
        if "categories" not in analysis or not analysis.get("categories") or len(analysis.get("categories", [])) == 0:
            # If categories are missing/empty, default to "No Concerns" if summary indicates no concerns
            summary_lower = analysis.get("consolidated_summary", "").lower()
            no_concern_indicators = [
                "no concern", "no issue", "no problem", "satisfactory",
                "going smoothly", "no current concern", "no active issue",
                "resolved", "everything is going", "no unresolved"
            ]
            if any(indicator in summary_lower for indicator in no_concern_indicators):
                analysis["categories"] = ["No Concerns"]
            else:
                # If we can't determine, log a warning and default to "No Concerns" as fallback
                print(f"Warning: No categories returned for user {user_id}, defaulting to 'No Concerns'. Summary: {analysis.get('consolidated_summary', 'N/A')[:80]}...")
                analysis["categories"] = ["No Concerns"]

        analysis["user_id"] = user_id
        return analysis

    except json.JSONDecodeError:
        return {
            "user_id": user_id,
            "error": "Failed to parse response"
        }
    except Exception as e:
        return {
            "user_id": user_id,
            "error": str(e)
        }


def analyze_all_users(grouped_data, filtered_data):
    """Analyze all users and return structured results"""
    results = []
    total_users = len(filtered_data)

    print(f"\nAnalyzing {total_users} users...")
    print("=" * 60)

    for idx, (user_id, data) in enumerate(filtered_data.items(), 1):
        print(f"[{idx}/{total_users}] Analyzing User {user_id}...", end=" ")

        analysis = analyze_user_feedback(user_id, data["notes"])
        results.append(analysis)

        if "error" in analysis:
            print(f"✗ Error: {analysis['error']}")
        else:
            categories_str = ", ".join(analysis.get('categories', []))
            print(f"✓ {categories_str}")

    return results


def generate_category_distribution(results):
    """Generate category distribution statistics"""
    all_categories = []
    for r in results:
        if "categories" in r:
            all_categories.extend(r.get("categories", []))

    category_counts = Counter(all_categories)
    total_occurrences = sum(category_counts.values())

    distribution = {}
    for category, count in category_counts.most_common():
        distribution[category] = {
            "count": count,
            "percentage": round((count / total_occurrences) * 100, 2)
        }

    return distribution


def generate_summary_report(results, filtered_data):
    """Generate comprehensive summary report"""
    print("\n" + "=" * 60)
    print("SUMMARY REPORT - USER FEEDBACK ANALYSIS")
    print("=" * 60)

    print(f"\nTotal Users Analyzed: {len(results)}")

    # Category distribution
    print("\nISSUE CATEGORY DISTRIBUTION:")
    print("-" * 60)
    distribution = generate_category_distribution(results)

    for i, (category, stats) in enumerate(distribution.items(), 1):
        print(f"{i}. {category}: {stats['count']} users ({stats['percentage']}%)")


def export_results(results, output_file="nps_analysis_results.json"):
    """Export results to JSON file"""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results exported to {output_file}")


def create_user_report(result):
    """Format individual user analysis report"""
    if "error" in result:
        return f"Error analyzing User {result['user_id']}: {result['error']}"

    report = f"""
╔════════════════════════════════════════════════════════════════╗
║                     USER ANALYSIS REPORT                       ║
╚════════════════════════════════════════════════════════════════╝

User ID: {result.get('user_id', 'N/A')}
Categories: {', '.join(result.get('categories', []))}

SUMMARY:
{result.get('consolidated_summary', 'N/A')}
"""
    return report


def export_to_csv(results, output_file: str):
    """Export results to CSV for easy viewing"""
    data = []
    for result in results:
        if "error" not in result:
            data.append({
                "user_id": result.get("user_id"),
                "categories": "; ".join(result.get("categories", [])),
                "summary": result.get("consolidated_summary")
            })

    df = pd.DataFrame(data)
    ensure_output_directory(os.path.dirname(output_file) or ".")
    df.to_csv(output_file, index=False)
    print(f"✓ CSV export to {output_file}")


def main(csv_file_path: Optional[str] = None):
    """Main execution function"""
    print("🚀 NPS ANALYSIS PIPELINE STARTED")
    print("=" * 60)

    if csv_file_path is None:
        csv_file_path = resolve_input_csv()
        print(f"Input CSV resolved to: {csv_file_path}")

    # Load data
    df = load_csv(csv_file_path)
    if df is None:
        return

    print(f"Columns found: {list(df.columns)}")

    # Group by user
    grouped_data = group_notes_by_user(df)
    print(f"✓ Grouped into {len(grouped_data)} unique users")

    # Get all users (filtering removed)
    filtered_data = filter_by_nps(grouped_data)
    print(f"✓ Processing {len(filtered_data)} users")

    if len(filtered_data) == 0:
        print("✗ No users found in data")
        return

    # Analyze all users
    results = analyze_all_users(grouped_data, filtered_data)

    # Generate reports
    generate_summary_report(results, filtered_data)

    # Export results to CSV only
    csv_stem = os.path.splitext(os.path.basename(csv_file_path))[0]
    output_csv_path = os.path.join(
        OUTPUT_DIRECTORY, f"{OUTPUT_CSV_BASENAME}_{csv_stem}.csv"
    )
    export_to_csv(results, output_csv_path)

    # Display sample user reports
    print("\n" + "=" * 60)
    print("SAMPLE USER REPORTS (First 5)")
    print("=" * 60)
    for result in results[:5]:
        print(create_user_report(result))

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()

