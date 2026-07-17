"""Course recommender demo -- multi-model picker.

Deploy-ready for Streamlit Community Cloud: reads only precomputed local
files (data/app_recommendations/*.parquet, results/comparison_common.csv,
data/course_catalog.csv, data/ratings_full.csv, figures/*.png). It never
fits a model or imports anything from src/models/ at runtime -- those are
heavy (TensorFlow, Surprise, scikit-learn) and this app doesn't need them.
See scripts/precompute_app_recommendations.py for how the parquet files
are generated (run once, offline, not at app startup).
"""

import os

import pandas as pd
import streamlit as st

APP_RECS_DIR = os.path.join("data", "app_recommendations")
COMPARISON_PATH = os.path.join("results", "comparison_common.csv")
COURSE_CATALOG_PATH = os.path.join("data", "course_catalog.csv")
RATINGS_PATH = os.path.join("data", "ratings_full.csv")
ENROLLMENT_CHART_PATH = os.path.join("figures", "user_enrollment_distribution.png")

DEFAULT_MODEL = "content_clustering"

MODEL_LABELS = {
    "content_clustering": "Content: Clustering (top performer)",
    "popularity_baseline": "Popularity baseline",
    "content_user_profile": "Content: User Profile",
    "content_course_similarity": "Content: Course Similarity",
    "cf_nmf": "Collaborative Filtering: NMF",
    "cf_classification": "Collaborative Filtering: Classification",
    "cf_ann": "Collaborative Filtering: Neural Net (cf_ann)",
    "cf_regression": "Collaborative Filtering: Regression",
    "cf_knn": "Collaborative Filtering: KNN",
}


@st.cache_data
def load_comparison_table() -> pd.DataFrame:
    df = pd.read_csv(COMPARISON_PATH)
    return df.sort_values("precision_at_10", ascending=False).reset_index(drop=True)


@st.cache_data
def load_course_catalog() -> pd.DataFrame:
    return pd.read_csv(COURSE_CATALOG_PATH)


@st.cache_data
def load_ratings() -> pd.DataFrame:
    return pd.read_csv(RATINGS_PATH)


@st.cache_data
def available_models() -> list:
    if not os.path.isdir(APP_RECS_DIR):
        return []
    names = [f[:-8] for f in os.listdir(APP_RECS_DIR) if f.endswith(".parquet")]
    ordered = [n for n in MODEL_LABELS if n in names]
    ordered += [n for n in names if n not in MODEL_LABELS]
    return ordered


@st.cache_data
def load_model_recommendations(model_name: str) -> pd.DataFrame:
    path = os.path.join(APP_RECS_DIR, f"{model_name}.parquet")
    return pd.read_parquet(path)


@st.cache_data
def user_list(ratings_df: pd.DataFrame) -> list:
    return sorted(ratings_df["user"].unique().tolist())


def main():
    st.set_page_config(page_title="Course Recommender", layout="wide")
    st.title("Course Recommender")
    st.caption(
        "Nine recommendation approaches, trained on the same 33,901-user IBM course-enrollment "
        "dataset, compared on a held-out ranking evaluation."
    )

    models = available_models()
    if not models:
        st.error(
            f"No precomputed recommendations found in {APP_RECS_DIR}/. "
            "Run `python -m scripts.precompute_app_recommendations` first."
        )
        st.stop()

    ratings_df = load_ratings()
    course_catalog = load_course_catalog()
    users = user_list(ratings_df)

    with st.sidebar:
        st.header("Controls")
        default_idx = models.index(DEFAULT_MODEL) if DEFAULT_MODEL in models else 0
        model_name = st.selectbox(
            "Model",
            options=models,
            index=default_idx,
            format_func=lambda m: MODEL_LABELS.get(m, m),
        )
        user_id = st.selectbox("User ID", options=users, index=0)
        top_n = st.slider("Number of recommendations", min_value=1, max_value=20, value=10)

    col_recs, col_history = st.columns([2, 1])

    with col_history:
        st.subheader(f"User {user_id}'s enrollment history")
        history = ratings_df[ratings_df["user"] == user_id].merge(
            course_catalog, how="left", left_on="item", right_on="COURSE_ID"
        )
        st.caption(f"{len(history)} course(s) enrolled")
        st.dataframe(
            history[["TITLE", "rating"]].rename(columns={"TITLE": "Course", "rating": "Rating"}),
            hide_index=True,
            use_container_width=True,
        )

    with col_recs:
        st.subheader(f"Top {top_n} recommendations -- {MODEL_LABELS.get(model_name, model_name)}")
        recs = load_model_recommendations(model_name)
        user_recs = recs[recs["USER"] == user_id].sort_values("SCORE", ascending=False).head(top_n)

        if user_recs.empty:
            st.info(
                "This model has no recommendations for this user "
                "(possible if the user falls outside the model's course/user coverage)."
            )
        else:
            display = user_recs[["TITLE", "COURSE_ID", "SCORE"]].rename(
                columns={"TITLE": "Course", "COURSE_ID": "Course ID", "SCORE": "Score"}
            )
            st.dataframe(display, hide_index=True, use_container_width=True)

    st.divider()

    st.header("Why content_clustering is the default")
    st.caption(
        "Tier-1 held-out evaluation: all models restricted to the same 124-course universe "
        "and the same 25,062 eval-eligible users. **Clustering is the only model that beats "
        "the popularity baseline** -- every collaborative-filtering model (cf_ann, "
        "cf_classification, cf_regression, cf_knn, cf_nmf) ranks below it."
    )
    comparison = load_comparison_table()
    display_comparison = comparison.copy()
    display_comparison["model"] = display_comparison["model"].map(lambda m: MODEL_LABELS.get(m, m))
    st.dataframe(
        display_comparison[
            ["model", "precision_at_10", "precision_at_20", "recall_at_10", "catalog_coverage"]
        ].rename(
            columns={
                "model": "Model",
                "precision_at_10": "Precision@10",
                "precision_at_20": "Precision@20",
                "recall_at_10": "Recall@10",
                "catalog_coverage": "Catalog Coverage",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("Why collaborative filtering underperforms here"):
        st.markdown(
            "**24.5% of users have exactly one course enrollment.** Collaborative filtering "
            "works by finding patterns across a user's *multiple* interactions -- for a quarter "
            "of this user base, there's nothing to learn from. This single-enrollment cohort is "
            "the concrete, visible reason every CF model in the comparison above underperforms "
            "a plain popularity ranking."
        )
        if os.path.exists(ENROLLMENT_CHART_PATH):
            st.image(ENROLLMENT_CHART_PATH, use_container_width=True)
        else:
            st.caption(f"(Chart not found at {ENROLLMENT_CHART_PATH})")


if __name__ == "__main__":
    main()
