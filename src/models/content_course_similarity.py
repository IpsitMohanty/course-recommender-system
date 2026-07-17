"""Content-based recommender using BoW course-to-course similarity.

Ported from lab_jupyter_content_course_similarity.ipynb (built on
lab_jupyter_fe_course_sim.ipynb's cosine-similarity method). The similarity
matrix is derived purely from course text content (title + description
BoW), not from any user rating, so fit(train_df) has nothing to learn from
train_df for this model -- it's a real method, kept for interface
uniformity, but a no-op.

`exclude` does double duty here, matching the original notebook's own
recommend-from-enrolled-courses design: it is both the seed set (find
courses similar to these) and the mask set (don't recommend these again).
A candidate course's score is the max similarity to any course in
`exclude`. Recommends from the full 307-course catalog (course_genres_df
covers every course), same universe as content_user_profile.
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.data import load_course_bows, load_course_genres
from src.models.base import RANDOM_STATE, Recommender

assert RANDOM_STATE == 123  # keep the global seed visible at the call site


class ContentCourseSimilarityRecommender(Recommender):
    def __init__(self, course_genres_df: pd.DataFrame = None, bows_df: pd.DataFrame = None):
        course_genres_df = course_genres_df if course_genres_df is not None else load_course_genres()
        bows_df = bows_df if bows_df is not None else load_course_bows()

        self.course_order = course_genres_df["COURSE_ID"].tolist()

        wide_bow = bows_df.pivot(index="doc_id", columns="token", values="bow").fillna(0)
        wide_bow = wide_bow.reindex(self.course_order).fillna(0)

        self.sim_matrix = cosine_similarity(wide_bow.to_numpy())
        self.id_idx_dict = {course_id: i for i, course_id in enumerate(self.course_order)}
        self.universe = set(self.course_order)  # full 307-course catalog

    def fit(self, train_df: pd.DataFrame) -> "ContentCourseSimilarityRecommender":
        return self  # no user-interaction data to learn from; see module docstring

    def recommend(self, user_id, k: int, exclude: set) -> pd.DataFrame:
        empty = pd.DataFrame(columns=["COURSE_ID", "SCORE"])
        if not exclude:
            return empty

        seed_indices = [self.id_idx_dict[c] for c in exclude if c in self.id_idx_dict]
        if not seed_indices:
            return empty

        scores = self.sim_matrix[seed_indices].max(axis=0)

        candidates = [
            (course_id, scores[idx])
            for course_id, idx in self.id_idx_dict.items()
            if course_id not in exclude
        ]
        if not candidates:
            return empty

        candidates.sort(key=lambda pair: pair[1], reverse=True)
        return pd.DataFrame(candidates[:k], columns=["COURSE_ID", "SCORE"])
