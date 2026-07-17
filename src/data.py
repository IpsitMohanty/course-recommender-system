"""Data loading and the canonical train/test holdout split.

The holdout split is built once (see build_holdout_split) and persisted to
data/splits/holdout.parquet so every model and the evaluation module score
against the exact same train/test division. It is a per-user split: each
user's own enrollments are divided into train and test, not a global random
split of rows, so every eval-eligible user has both train history (for
fit()) and held-out test courses (for scoring recommend()).
"""

import os

import numpy as np
import pandas as pd

RATINGS_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-ML0321EN-Coursera/labs/v2/module_3/ratings.csv"
COURSE_GENRE_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/course_genre.csv"

RANDOM_STATE = 123
TEST_FRACTION = 0.20
MIN_ENROLLMENTS_FOR_EVAL = 5  # users below this get all rows in train, none in test

SPLIT_PATH = os.path.join("data", "splits", "holdout.parquet")


def load_ratings() -> pd.DataFrame:
    """Raw module_3 ratings: columns user, item, rating."""
    return pd.read_csv(RATINGS_URL)


def load_course_genres() -> pd.DataFrame:
    """Course genre matrix: COURSE_ID, TITLE, then one binary column per genre."""
    return pd.read_csv(COURSE_GENRE_URL)


def load_course_bows(path: str = os.path.join("data", "courses_bows.csv")) -> pd.DataFrame:
    """BoW features built in lab_jupyter_fe_bow_solution.ipynb: doc_index, doc_id, token, bow."""
    return pd.read_csv(path)


def build_holdout_split(ratings_df: pd.DataFrame = None) -> pd.DataFrame:
    """Build the canonical per-user holdout split.

    For each user with >= MIN_ENROLLMENTS_FOR_EVAL enrollments, holds out
    max(1, round(TEST_FRACTION * n_enrolled)) of their ratings as test, the
    rest as train. Users below the threshold get all rows in train (eval
    excludes them, but models can still train on their ratings).

    Returns ratings_df with an added 'split' column ('train' or 'test').
    """
    if ratings_df is None:
        ratings_df = load_ratings()

    df = ratings_df.reset_index(drop=True)
    rng = np.random.default_rng(RANDOM_STATE)

    split_col = np.full(len(df), "train", dtype=object)

    for _, group in df.groupby("user"):
        n = len(group)
        if n < MIN_ENROLLMENTS_FOR_EVAL:
            continue
        n_test = max(1, round(TEST_FRACTION * n))
        test_positions = rng.choice(group.index.to_numpy(), size=n_test, replace=False)
        split_col[test_positions] = "test"

    df["split"] = split_col
    return df


def save_split(split_df: pd.DataFrame, path: str = SPLIT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    split_df.to_parquet(path, index=False)


def load_split(path: str = SPLIT_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def get_or_build_split(path: str = SPLIT_PATH) -> pd.DataFrame:
    """Load the persisted split if present, else build, persist, and return it."""
    if os.path.exists(path):
        return load_split(path)
    split_df = build_holdout_split()
    save_split(split_df, path)
    return split_df
