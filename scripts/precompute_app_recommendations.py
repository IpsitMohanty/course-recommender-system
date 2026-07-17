"""One-time precompute script for the Streamlit app.

Fits all 9 models (8 recommenders + popularity baseline) on the FULL
enrollment history and generates every user's top-20 recommendations,
masking each user's complete history (exclude = full history).

This is deliberately different from src/evaluation.py, which fits on the
train split only and masks train-only history, to score against a
held-out test set. This script has no held-out set to protect -- it's
building what the deployed app actually serves to real users, so it uses
every enrollment on record.

The deployed app must NOT fit models at request time (Streamlit Community
Cloud's memory limits, cf_ann's Keras model especially) -- it only ever
reads the parquet files this script writes to data/app_recommendations/.

Run manually, not at app startup:
    python -m scripts.precompute_app_recommendations
"""

import os

import pandas as pd

from src.data import load_course_bows, load_course_genres, load_ratings
from src.models.cf_ann import CFAnnRecommender
from src.models.cf_classification import CFClassificationRecommender
from src.models.cf_knn import CFKnnRecommender
from src.models.cf_nmf import CFNmfRecommender
from src.models.cf_regression import CFRegressionRecommender
from src.models.content_clustering import ContentClusteringRecommender
from src.models.content_course_similarity import ContentCourseSimilarityRecommender
from src.models.content_user_profile import ContentUserProfileRecommender
from src.models.popularity_baseline import PopularityBaselineRecommender

RANDOM_STATE = 123
TOP_K = 20
OUTPUT_DIR = os.path.join("data", "app_recommendations")

# Full-data cf_ann embeddings get their OWN paths, distinct from src/evaluation.py's
# train-only ones (data/user_embeddings_train.csv / data/course_embeddings_train.csv).
# Never let this script's CFAnnRecommender() fall back to the default path, or it
# would silently overwrite the eval artifacts with full-data embeddings.
FULL_USER_EMBEDDINGS_PATH = os.path.join("data", "user_embeddings_full.csv")
FULL_COURSE_EMBEDDINGS_PATH = os.path.join("data", "course_embeddings_full.csv")


def fit_all_models(full_df: pd.DataFrame) -> dict:
    course_genres_df = load_course_genres()
    bows_df = load_course_bows()

    models = {}

    print("Fitting cf_ann on full data (keystone -- exports full-data embeddings for the two models below)...")
    cf_ann = CFAnnRecommender(
        user_embeddings_path=FULL_USER_EMBEDDINGS_PATH,
        course_embeddings_path=FULL_COURSE_EMBEDDINGS_PATH,
    )
    cf_ann.fit(full_df)
    models["cf_ann"] = cf_ann

    print("Fitting content_user_profile...")
    models["content_user_profile"] = ContentUserProfileRecommender(course_genres_df).fit(full_df)

    print("Fitting content_course_similarity...")
    models["content_course_similarity"] = ContentCourseSimilarityRecommender(course_genres_df, bows_df).fit(full_df)

    print("Fitting content_clustering...")
    models["content_clustering"] = ContentClusteringRecommender(course_genres_df).fit(full_df)

    print("Fitting cf_classification (uses cf_ann's full-data embeddings, in-memory)...")
    models["cf_classification"] = CFClassificationRecommender(
        user_emb_df=cf_ann.user_embeddings_df, course_emb_df=cf_ann.course_embeddings_df
    ).fit(full_df)

    print("Fitting cf_regression (uses cf_ann's full-data embeddings, in-memory)...")
    models["cf_regression"] = CFRegressionRecommender(
        user_emb_df=cf_ann.user_embeddings_df, course_emb_df=cf_ann.course_embeddings_df
    ).fit(full_df)

    print("Fitting cf_knn...")
    models["cf_knn"] = CFKnnRecommender().fit(full_df)

    print("Fitting cf_nmf...")
    models["cf_nmf"] = CFNmfRecommender().fit(full_df)

    print("Fitting popularity_baseline...")
    models["popularity_baseline"] = PopularityBaselineRecommender().fit(full_df)

    return models


def generate_and_save(name: str, model, all_users: list, history_by_user: dict, course_titles: pd.DataFrame) -> int:
    rows = []
    for user_id in all_users:
        exclude = history_by_user.get(user_id, set())
        recs = model.recommend(user_id, TOP_K, exclude)
        for course_id, score in zip(recs["COURSE_ID"], recs["SCORE"]):
            rows.append((user_id, course_id, score))

    df = pd.DataFrame(rows, columns=["USER", "COURSE_ID", "SCORE"])
    df = df.merge(course_titles, how="left", on="COURSE_ID")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{name}.parquet")
    df.to_parquet(out_path, index=False)
    print(f"  Saved {df.shape[0]} rows ({df['USER'].nunique()} users) -> {out_path}")
    return df.shape[0]


def main():
    full_df = load_ratings()
    all_users = full_df["user"].unique().tolist()
    history_by_user = full_df.groupby("user")["item"].apply(set).to_dict()

    course_genres_df = load_course_genres()
    course_titles = course_genres_df[["COURSE_ID", "TITLE"]]

    print(f"Full dataset: {len(full_df)} ratings, {len(all_users)} users")

    models = fit_all_models(full_df)

    print(f"\nGenerating top-{TOP_K} recommendations (exclude = full history) "
          f"for {len(all_users)} users, {len(models)} models...")
    summary = {}
    for name, model in models.items():
        print(f"  {name}...")
        summary[name] = generate_and_save(name, model, all_users, history_by_user, course_titles)

    print("\nDone. Row counts per model:")
    for name, count in summary.items():
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
