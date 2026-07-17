"""Evaluation and model-comparison module.

Fits every model in src/models/ on the train side of the canonical holdout
split (src/data.py), scores each one's top-20 recommendations against each
user's held-out test enrollments, and produces two comparison tables:

- Tier 1 (results/comparison_common.csv): every model restricted to the
  shared course universe and the common eval-user set every model can
  cover. This is the apples-to-apples table.
- Tier 2 (results/comparison_native.csv): every model on its own full
  universe and its own full coverable user set. Shows reach/coverage
  differences, not for cross-model ranking.

Run directly: `python -m src.evaluation`
"""

import os

import numpy as np
import pandas as pd

from src.data import get_or_build_split, load_course_bows, load_course_genres
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
K_VALUES = (10, 20)
TOP_K = 20
RESULTS_DIR = "results"


# ---------------------------------------------------------------------------
# Setup: split, eval context, model fitting
# ---------------------------------------------------------------------------

def build_eval_context(split_df: pd.DataFrame):
    """Return (train_df, eval_users, train_by_user, test_by_user).

    eval_users = users with >=1 test row (the five-core eval-eligible set
    from the canonical split). train_by_user is what every model's
    recommend() call masks via `exclude`; test_by_user is the relevant set
    precision/recall are scored against.
    """
    train_df = split_df[split_df["split"] == "train"]
    test_df = split_df[split_df["split"] == "test"]

    train_by_user = train_df.groupby("user")["item"].apply(set).to_dict()
    test_by_user = test_df.groupby("user")["item"].apply(set).to_dict()
    eval_users = list(test_by_user.keys())

    return train_df, eval_users, train_by_user, test_by_user


def fit_all_models(train_df: pd.DataFrame) -> dict:
    """Fit every model on train_df only. cf_ann must fit first: cf_classification
    and cf_regression load its train-only embeddings from disk."""
    course_genres_df = load_course_genres()
    bows_df = load_course_bows()

    models = {}

    print("Fitting cf_ann (keystone -- exports train-only embeddings for the two models below)...")
    models["cf_ann"] = CFAnnRecommender().fit(train_df)

    print("Fitting content_user_profile...")
    models["content_user_profile"] = ContentUserProfileRecommender(course_genres_df).fit(train_df)

    print("Fitting content_course_similarity...")
    models["content_course_similarity"] = ContentCourseSimilarityRecommender(course_genres_df, bows_df).fit(train_df)

    print("Fitting content_clustering...")
    models["content_clustering"] = ContentClusteringRecommender(course_genres_df).fit(train_df)

    print("Fitting cf_classification (loads cf_ann's train-only embeddings)...")
    models["cf_classification"] = CFClassificationRecommender().fit(train_df)

    print("Fitting cf_regression (loads cf_ann's train-only embeddings)...")
    models["cf_regression"] = CFRegressionRecommender().fit(train_df)

    print("Fitting cf_knn...")
    models["cf_knn"] = CFKnnRecommender().fit(train_df)

    print("Fitting cf_nmf...")
    models["cf_nmf"] = CFNmfRecommender().fit(train_df)

    print("Fitting popularity_baseline...")
    models["popularity_baseline"] = PopularityBaselineRecommender().fit(train_df)

    return models


# ---------------------------------------------------------------------------
# Recommendation generation
# ---------------------------------------------------------------------------

def generate_recommendations(model, eval_users, train_by_user, k: int = TOP_K) -> dict:
    """{user_id: [course_id, ...]} ranked recommendation lists, one per eval user.
    Users the model can't produce a recommendation for are simply absent."""
    recs = {}
    for user_id in eval_users:
        exclude = train_by_user.get(user_id, set())
        result = model.recommend(user_id, k, exclude)
        if not result.empty:
            recs[user_id] = result["COURSE_ID"].tolist()
    return recs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_at_k(rec_course_ids: list, relevant: set, k: int) -> float:
    top_k = rec_course_ids[:k]
    hits = sum(1 for c in top_k if c in relevant)
    return hits / k


def recall_at_k(rec_course_ids: list, relevant: set, k: int) -> float:
    if not relevant:
        return np.nan
    top_k = rec_course_ids[:k]
    hits = sum(1 for c in top_k if c in relevant)
    return hits / len(relevant)


def catalog_coverage(recs_by_user: dict, universe_size: int) -> float:
    if universe_size == 0:
        return np.nan
    recommended_courses = set()
    for course_ids in recs_by_user.values():
        recommended_courses.update(course_ids)
    return len(recommended_courses) / universe_size


def score_model(model_name: str, recs_by_user: dict, test_by_user: dict, universe_size: int, total_eval_users: int) -> dict:
    """Aggregate precision/recall/coverage over the users recs_by_user covers."""
    p10s, p20s, r10s = [], [], []
    for user_id, course_ids in recs_by_user.items():
        relevant = test_by_user[user_id]
        p10s.append(precision_at_k(course_ids, relevant, 10))
        p20s.append(precision_at_k(course_ids, relevant, 20))
        r10s.append(recall_at_k(course_ids, relevant, 10))

    users_evaluated = len(recs_by_user)
    users_excluded = total_eval_users - users_evaluated

    return {
        "model": model_name,
        "universe_size": universe_size,
        "users_evaluated": users_evaluated,
        "users_excluded": users_excluded,
        "precision_at_10": np.mean(p10s) if p10s else np.nan,
        "precision_at_20": np.mean(p20s) if p20s else np.nan,
        "recall_at_10": np.mean(r10s) if r10s else np.nan,
        "catalog_coverage": catalog_coverage(recs_by_user, universe_size),
    }


# ---------------------------------------------------------------------------
# Two-tier comparison
# ---------------------------------------------------------------------------

def restrict_recs_to_universe(recs_by_user: dict, allowed_courses: set) -> dict:
    return {
        user_id: [c for c in course_ids if c in allowed_courses]
        for user_id, course_ids in recs_by_user.items()
    }


def run_comparison(models: dict, eval_users: list, train_by_user: dict, test_by_user: dict):
    total_eval_users = len(eval_users)

    print(f"\nGenerating top-{TOP_K} recommendations for {total_eval_users} eval users, {len(models)} models...")
    raw_recs = {}
    for name, model in models.items():
        print(f"  {name}...")
        raw_recs[name] = generate_recommendations(model, eval_users, train_by_user, TOP_K)

    # ---- Tier 2: native universe, native user coverage ----
    tier2_rows = []
    for name, model in models.items():
        universe_size = len(model.universe)
        row = score_model(name, raw_recs[name], test_by_user, universe_size, total_eval_users)
        tier2_rows.append(row)
    tier2_df = pd.DataFrame(tier2_rows)

    # ---- Tier 1: shared universe, common user set ----
    common_universe = set.intersection(*[m.universe for m in models.values()])
    common_users = set(eval_users)
    for name in models:
        common_users &= set(raw_recs[name].keys())

    print(f"\nTier 1 shared universe size: {len(common_universe)} courses "
          f"(expected ~126; reporting actual, not assuming)")
    print(f"Tier 1 common user set: {len(common_users)} of {total_eval_users} eval users "
          f"covered by every model")

    tier1_rows = []
    for name in models:
        restricted = restrict_recs_to_universe(raw_recs[name], common_universe)
        restricted = {u: c for u, c in restricted.items() if u in common_users}
        row = score_model(name, restricted, test_by_user, len(common_universe), len(common_users))
        tier1_rows.append(row)
    tier1_df = pd.DataFrame(tier1_rows)

    return tier1_df, tier2_df, common_universe, common_users


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(df: pd.DataFrame, title: str):
    ordered = df.sort_values("precision_at_10", ascending=False).reset_index(drop=True)
    display = ordered.copy()
    for col in ["precision_at_10", "precision_at_20", "recall_at_10", "catalog_coverage"]:
        display[col] = display[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "n/a")
    print(f"\n=== {title} (sorted by Precision@10) ===")
    print(display.to_string(index=False))
    return ordered


def main():
    split_df = get_or_build_split()
    train_df, eval_users, train_by_user, test_by_user = build_eval_context(split_df)
    print(f"Eval-eligible users: {len(eval_users)}")

    models = fit_all_models(train_df)

    tier1_df, tier2_df, common_universe, common_users = run_comparison(
        models, eval_users, train_by_user, test_by_user
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    tier1_sorted = print_table(tier1_df, "Tier 1 -- common universe, common users")
    tier2_sorted = print_table(tier2_df, "Tier 2 -- native universe, native users")

    tier1_sorted.to_csv(os.path.join(RESULTS_DIR, "comparison_common.csv"), index=False)
    tier2_sorted.to_csv(os.path.join(RESULTS_DIR, "comparison_native.csv"), index=False)

    baseline_rank = tier1_sorted.reset_index(drop=True)
    baseline_pos = baseline_rank.index[baseline_rank["model"] == "popularity_baseline"].tolist()
    if baseline_pos:
        pos = baseline_pos[0] + 1
        n = len(baseline_rank)
        beaten_by = baseline_rank.iloc[:pos - 1]["model"].tolist()
        print(f"\npopularity_baseline ranks #{pos} of {n} on Tier-1 Precision@10.")
        if pos == 1:
            print("No trained model beat the popularity baseline.")
        else:
            print(f"Models that beat it: {beaten_by}")

    print(f"\nSaved results/comparison_common.csv and results/comparison_native.csv")
    return tier1_sorted, tier2_sorted


if __name__ == "__main__":
    main()
