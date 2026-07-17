"""Popularity baseline -- the reference line every trained model should beat.

Ignores user identity entirely: recommends the globally most-enrolled
(train-split) courses to everyone, masking only `exclude`. SCORE = global
enrollment count among train rows. If a trained model can't outperform
this, that's the most important thing the comparison table reveals.
"""

import pandas as pd

from src.models.base import RANDOM_STATE, Recommender

assert RANDOM_STATE == 123  # keep the global seed visible at the call site


class PopularityBaselineRecommender(Recommender):
    def __init__(self):
        self._ranked_courses = []  # list of (course_id, count), sorted descending

    def fit(self, train_df: pd.DataFrame) -> "PopularityBaselineRecommender":
        counts = train_df.groupby("item").size().sort_values(ascending=False)
        self._ranked_courses = list(counts.items())
        self.universe = set(counts.index)
        return self

    def recommend(self, user_id, k: int, exclude: set) -> pd.DataFrame:
        empty = pd.DataFrame(columns=["COURSE_ID", "SCORE"])

        candidates = [(course_id, score) for course_id, score in self._ranked_courses if course_id not in exclude]
        if not candidates:
            return empty

        return pd.DataFrame(candidates[:k], columns=["COURSE_ID", "SCORE"])
