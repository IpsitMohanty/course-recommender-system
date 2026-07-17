"""Non-negative matrix factorization collaborative filtering recommender (Surprise).

Ported from lab_jupyter_cf_nmf.ipynb. Fits on train_df only via Surprise's
build_full_trainset() (no internal Surprise-side train/test split) -- unlike
the earlier prediction-file version of this model, which lost ~2,547 users
to Surprise's own unseeded random split, every user with train rows gets
factor-matrix coverage here.

Scores are vectorized via the factor matrices (algo.pu, algo.qi) rather
than per-pair .predict() calls. algo.pu/algo.qi are indexed by Surprise's
internal ids -- mapped back to raw ids via trainset.to_raw_uid()/
to_raw_iid() before use, or recommendations would be silently misindexed.
"""

import pandas as pd
from surprise import NMF, Dataset, Reader

from src.models.base import RANDOM_STATE, Recommender

assert RANDOM_STATE == 123  # keep the global seed visible at the call site


class CFNmfRecommender(Recommender):
    def __init__(self, n_factors: int = 32):
        self.n_factors = n_factors
        self.algo = None
        self.trainset = None
        self.user_ids = []
        self.course_ids = []
        self.user_row = {}
        self._score_matrix = None

    def fit(self, train_df: pd.DataFrame) -> "CFNmfRecommender":
        rating_scale = (train_df["rating"].min(), train_df["rating"].max())
        reader = Reader(rating_scale=rating_scale)
        surprise_data = Dataset.load_from_df(train_df[["user", "item", "rating"]], reader)
        self.trainset = surprise_data.build_full_trainset()

        self.algo = NMF(
            n_factors=self.n_factors,
            init_low=0.5,
            init_high=5.0,
            random_state=RANDOM_STATE,
            verbose=False,
        )
        self.algo.fit(self.trainset)

        self.user_ids = [self.trainset.to_raw_uid(i) for i in range(self.trainset.n_users)]
        self.course_ids = [self.trainset.to_raw_iid(i) for i in range(self.trainset.n_items)]
        self.user_row = {u: i for i, u in enumerate(self.user_ids)}
        self.universe = set(self.course_ids)

        self._score_matrix = self.algo.pu @ self.algo.qi.T
        return self

    def recommend(self, user_id, k: int, exclude: set) -> pd.DataFrame:
        empty = pd.DataFrame(columns=["COURSE_ID", "SCORE"])

        u_idx = self.user_row.get(user_id)
        if u_idx is None:
            return empty

        scores = self._score_matrix[u_idx]
        candidates = [
            (course_id, scores[idx])
            for idx, course_id in enumerate(self.course_ids)
            if course_id not in exclude
        ]
        if not candidates:
            return empty

        candidates.sort(key=lambda pair: pair[1], reverse=True)
        return pd.DataFrame(candidates[:k], columns=["COURSE_ID", "SCORE"])
