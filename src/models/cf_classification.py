"""Classification-based rating-mode recommender using cf_ann's embeddings.

Ported from lab_jupyter_cf_classification_w_embeddings.ipynb. Predicts a
3-class rating target (module_3 ratings are {3,4,5}, not the binary
audit/complete scale the original lab assumed) from the element-wise sum
of a user's and course's cf_ann embeddings. SCORE = expected rating =
predict_proba() columns dotted with their real rating values (read off
label_encoder.classes_[model.classes_], not hardcoded).

Depends on cf_ann's train-only embeddings (data/user_embeddings_train.csv /
data/course_embeddings_train.csv) -- CFAnnRecommender.fit() must run first
in any orchestration. Loading is explicit and fails loudly if those files
are missing, rather than silently falling back to the old full-data
data/user_embeddings.csv / data/course_embeddings.csv.
"""

import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from src.models.base import RANDOM_STATE, Recommender
from src.models.cf_ann import COURSE_EMBEDDINGS_PATH, USER_EMBEDDINGS_PATH

assert RANDOM_STATE == 123  # keep the global seed visible at the call site


def _require_train_only_embeddings():
    for path in (USER_EMBEDDINGS_PATH, COURSE_EMBEDDINGS_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. src.models.cf_ann.CFAnnRecommender.fit() must run "
                "first -- this model depends on its train-only embeddings, not the old "
                "full-data data/user_embeddings.csv / data/course_embeddings.csv."
            )


class CFClassificationRecommender(Recommender):
    def __init__(self, user_emb_df: pd.DataFrame = None, course_emb_df: pd.DataFrame = None):
        if user_emb_df is None or course_emb_df is None:
            _require_train_only_embeddings()
        self.user_emb_df = user_emb_df if user_emb_df is not None else pd.read_csv(USER_EMBEDDINGS_PATH)
        self.course_emb_df = course_emb_df if course_emb_df is not None else pd.read_csv(COURSE_EMBEDDINGS_PATH)

        self.u_features = [c for c in self.user_emb_df.columns if c != "user"]
        self.c_features = [c for c in self.course_emb_df.columns if c != "item"]

        self.user_ids = self.user_emb_df["user"].tolist()
        self.course_ids = self.course_emb_df["item"].tolist()
        self.user_matrix = self.user_emb_df[self.u_features].to_numpy(dtype="float32")
        self.course_matrix = self.course_emb_df[self.c_features].to_numpy(dtype="float32")
        self.user_row = {u: i for i, u in enumerate(self.user_ids)}
        self.universe = set(self.course_ids)

        self.model = None
        self.label_encoder = None
        self._score_matrix = None  # (n_users x n_courses) expected-rating grid, cached at fit time

    def fit(self, train_df: pd.DataFrame) -> "CFClassificationRecommender":
        merged = train_df.merge(self.user_emb_df, how="inner", on="user").merge(
            self.course_emb_df, how="inner", left_on="item", right_on="item"
        )
        X = merged[self.u_features].to_numpy() + merged[self.c_features].to_numpy()
        y_raw = merged["rating"]

        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y_raw.to_numpy().ravel())

        self.model = RandomForestClassifier(max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
        self.model.fit(X, y)

        interaction = self.user_matrix[:, None, :] + self.course_matrix[None, :, :]
        interaction = interaction.reshape(-1, interaction.shape[-1])
        probas = self.model.predict_proba(interaction)
        rating_values_per_column = self.label_encoder.classes_[self.model.classes_]
        expected_rating = probas @ rating_values_per_column
        self._score_matrix = expected_rating.reshape(len(self.user_ids), len(self.course_ids))

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
