"""Content-based recommender using user genre-profile vectors.

Ported from lab_jupyter_content_user_profile.ipynb. Method: build each
user's profile as sum(rating * genre_vector) over their train-split
enrollments, then score every course by the dot product of the user's
profile vector and that course's genre vector. Recommends from the full
307-course catalog (course_genres_df covers every course, not just the ones
that appear in ratings.csv) -- this is one of the two 307-universe models
in this project, unlike the collaborative-filtering models which only know
the ~126 courses that have ratings.
"""

import numpy as np
import pandas as pd

from src.data import load_course_genres
from src.models.base import RANDOM_STATE, Recommender

assert RANDOM_STATE == 123  # keep the global seed visible at the call site


class ContentUserProfileRecommender(Recommender):
    def __init__(self, course_genres_df: pd.DataFrame = None):
        self.course_genres_df = course_genres_df if course_genres_df is not None else load_course_genres()
        self.course_order = self.course_genres_df["COURSE_ID"].tolist()
        self.genre_cols = list(self.course_genres_df.columns[2:])
        self.genre_matrix = self.course_genres_df[self.genre_cols].to_numpy()
        self.universe = set(self.course_order)  # full 307-course catalog

        self._profile_by_user = {}  # user_id -> profile vector (len(genre_cols),)

    def fit(self, train_df: pd.DataFrame) -> "ContentUserProfileRecommender":
        rating_matrix = (
            train_df.pivot_table(index="user", columns="item", values="rating", aggfunc="max")
            .reindex(columns=self.course_order, fill_value=0)
            .fillna(0)
        )
        profile_matrix = rating_matrix.to_numpy() @ self.genre_matrix  # (n_users x n_genres)

        self._profile_by_user = {
            user_id: profile_matrix[i] for i, user_id in enumerate(rating_matrix.index)
        }
        return self

    def recommend(self, user_id, k: int, exclude: set) -> pd.DataFrame:
        empty = pd.DataFrame(columns=["COURSE_ID", "SCORE"])

        profile_vector = self._profile_by_user.get(user_id)
        if profile_vector is None:
            return empty  # user had no train-split enrollments (or wasn't in train_df at all)

        scores = self.genre_matrix @ profile_vector  # (n_courses,)

        candidates = [
            (course_id, scores[i])
            for i, course_id in enumerate(self.course_order)
            if course_id not in exclude
        ]
        if not candidates:
            return empty

        candidates.sort(key=lambda pair: pair[1], reverse=True)
        top_k = candidates[:k]
        return pd.DataFrame(top_k, columns=["COURSE_ID", "SCORE"])
