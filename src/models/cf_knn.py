"""Item-based KNN collaborative filtering recommender (Surprise).

Ported from lab_jupyter_cf_knn.ipynb. Item-based (not user-based, despite
the notebook's own worked example) because this dataset has ~33,901 users
vs. ~126 courses -- item-based means a ~126x126 similarity matrix instead
of a ~33,901x33,901 one.

Fits on train_df only via Surprise's build_full_trainset() (no internal
Surprise-side train/test split -- the real held-out set is the caller's
canonical split, not something this model should re-split on its own).
"""

import pandas as pd
from surprise import Dataset, KNNBasic, Reader

from src.models.base import RANDOM_STATE, Recommender

assert RANDOM_STATE == 123  # keep the global seed visible at the call site


class CFKnnRecommender(Recommender):
    def __init__(self):
        self.algo = None
        self.trainset = None
        self.course_ids = []

    def fit(self, train_df: pd.DataFrame) -> "CFKnnRecommender":
        rating_scale = (train_df["rating"].min(), train_df["rating"].max())
        reader = Reader(rating_scale=rating_scale)
        surprise_data = Dataset.load_from_df(train_df[["user", "item", "rating"]], reader)
        self.trainset = surprise_data.build_full_trainset()

        self.algo = KNNBasic(sim_options={"name": "cosine", "user_based": False}, verbose=False)
        self.algo.fit(self.trainset)

        self.course_ids = [self.trainset.to_raw_iid(i) for i in range(self.trainset.n_items)]
        self.universe = set(self.course_ids)
        return self

    def recommend(self, user_id, k: int, exclude: set) -> pd.DataFrame:
        empty = pd.DataFrame(columns=["COURSE_ID", "SCORE"])

        try:
            self.trainset.to_inner_uid(user_id)
        except ValueError:
            return empty  # user had no train-split rows (or wasn't passed the raw id Surprise expects)

        candidates = [c for c in self.course_ids if c not in exclude]
        if not candidates:
            return empty

        scored = [(c, self.algo.predict(user_id, c).est) for c in candidates]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return pd.DataFrame(scored[:k], columns=["COURSE_ID", "SCORE"])
