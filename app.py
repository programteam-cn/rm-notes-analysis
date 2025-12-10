import io
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from main import (
    REQUIRED_COLUMNS,
    analyze_user_feedback,
    filter_by_nps,
    generate_category_distribution,
    group_notes_by_user,
)

# Load environment variables from .env file
load_dotenv()


st.set_page_config(page_title="NPS Notes Analysis", layout="wide")

st.title("RM Notes Analysis")
st.write(
    "Upload an RM notes CSV and run the analysis. "
    "The prompt in `prompt.txt` controls summarization and categories."
)


def ensure_api_key():
    """Get API key from .env file and place into env for lazy client init."""
    key = os.getenv("OPENAI_API_KEY")
    if key:
        os.environ["OPENAI_API_KEY"] = key.strip()
        return key.strip()
    return None


def validate_columns(df: pd.DataFrame):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return False
    return True


def run_analysis(df: pd.DataFrame):
    grouped = group_notes_by_user(df)
    filtered = filter_by_nps(grouped)
    if not filtered:
        st.warning("No users found in this file.")
        return []

    progress = st.progress(0.0, text="Analyzing users...")
    results = []
    total = len(filtered)
    for idx, (user_id, data) in enumerate(filtered.items(), 1):
        result = analyze_user_feedback(user_id, data["notes"])
        results.append(result)
        progress.progress(idx / total, text=f"Analyzed {idx}/{total} users")
    progress.empty()
    return results


def show_results(results):
    if not results:
        return

    df_results = pd.DataFrame(
        [
            {
                "user_id": r.get("user_id"),
                "categories": "; ".join(r.get("categories", []))
                if "categories" in r
                else "",
                "summary": r.get("consolidated_summary", r.get("error", "")),
            }
            for r in results
            if "error" not in r
        ]
    )

    st.subheader("Analysis Output")
    st.dataframe(df_results, use_container_width=True)

    distribution = generate_category_distribution(results)
    if distribution:
        dist_df = pd.DataFrame(
            [
                {"category": cat, "count": stats["count"], "percentage": stats["percentage"]}
                for cat, stats in distribution.items()
            ]
        )
        st.subheader("Category Distribution")
        st.dataframe(dist_df, use_container_width=True)

    csv_buf = io.StringIO()
    df_results.to_csv(csv_buf, index=False)
    st.download_button(
        label="Download CSV",
        data=csv_buf.getvalue(),
        file_name="analysis_results.csv",
        mime="text/csv",
    )


api_key = ensure_api_key()
uploaded = st.file_uploader("Upload RM notes CSV", type=["csv"])

if st.button("Run Analysis", type="primary"):
    if not api_key:
        st.error("Please provide an OpenAI API key.")
    elif not uploaded:
        st.error("Please upload a CSV file.")
    else:
        try:
            df_input = pd.read_csv(uploaded)
            if validate_columns(df_input):
                results = run_analysis(df_input)
                show_results(results)
        except Exception as exc:
            st.error(f"Failed to process file: {exc}")

