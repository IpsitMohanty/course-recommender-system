"""Neural collaborative filtering (embedding-based) recommender.

Ported from lab_jupyter_cf_ann.ipynb. RecommenderNet learns latent user and
course embeddings by predicting (min-max scaled) ratings from a user index
and a course index.

This is the keystone model in this package: cf_classification.py and
cf_regression.py consume the embeddings this model learns. fit() must run
(and export its embeddings) before those two are fit, or they'd either
fail to find the file or -- worse -- silently fall back to stale full-data
embeddings from an earlier stage. USER_EMBEDDINGS_PATH / COURSE_EMBEDDINGS_PATH
are deliberately different filenames from the old data/user_embeddings.csv /
data/course_embeddings.csv (full-data, notebook-era) so there is no way to
accidentally read the wrong ones.
"""

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.base import RANDOM_STATE, Recommender

assert RANDOM_STATE == 123  # keep the global seed visible at the call site

USER_EMBEDDINGS_PATH = os.path.join("data", "user_embeddings_train.csv")
COURSE_EMBEDDINGS_PATH = os.path.join("data", "course_embeddings_train.csv")


class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_items, embedding_size=16, **kwargs):
        super().__init__(**kwargs)
        self.user_embedding_layer = layers.Embedding(
            input_dim=num_users,
            output_dim=embedding_size,
            name="user_embedding_layer",
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(input_dim=num_users, output_dim=1, name="user_bias")
        self.item_embedding_layer = layers.Embedding(
            input_dim=num_items,
            output_dim=embedding_size,
            name="item_embedding_layer",
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.item_bias = layers.Embedding(input_dim=num_items, output_dim=1, name="item_bias")

    def call(self, inputs):
        user_vector = self.user_embedding_layer(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        item_vector = self.item_embedding_layer(inputs[:, 1])
        item_bias = self.item_bias(inputs[:, 1])
        dot_user_item = tf.tensordot(user_vector, item_vector, 2)
        x = dot_user_item + user_bias + item_bias
        return tf.nn.relu(x)


class CFAnnRecommender(Recommender):
    def __init__(
        self,
        embedding_size: int = 16,
        epochs: int = 10,
        batch_size: int = 64,
        user_embeddings_path: str = None,
        course_embeddings_path: str = None,
    ):
        self.embedding_size = embedding_size
        self.epochs = epochs
        self.batch_size = batch_size
        # Overridable so a full-data fit (e.g. the app precompute script) can export
        # to a different path than the train-only fit (src/evaluation.py's default),
        # instead of silently overwriting the eval artifacts. Resolved here (not as a
        # literal default-argument value) so it still respects a monkeypatched module
        # constant at call time -- default-argument values are frozen at import time.
        self.user_embeddings_path = user_embeddings_path or USER_EMBEDDINGS_PATH
        self.course_embeddings_path = course_embeddings_path or COURSE_EMBEDDINGS_PATH

        self.model = None
        self.user_id2idx = {}
        self.course_id2idx = {}
        self.user_embeddings = None
        self.course_embeddings = None
        self.user_bias = None
        self.course_bias = None
        self.user_embeddings_df = None  # in-memory copy of what gets exported, for
        self.course_embeddings_df = None  # callers that want to skip the CSV round-trip

    def fit(self, train_df: pd.DataFrame) -> "CFAnnRecommender":
        # tf.random.set_seed() alone is not sufficient for reproducibility across
        # independently-constructed model instances in the same process (verified
        # empirically -- two fits with only tf.random.set_seed still diverged).
        # set_random_seed() seeds Python's random, NumPy, and TF together.
        tf.keras.utils.set_random_seed(RANDOM_STATE)

        user_list = train_df["user"].unique().tolist()
        course_list = train_df["item"].unique().tolist()
        self.user_id2idx = {u: i for i, u in enumerate(user_list)}
        self.course_id2idx = {c: i for i, c in enumerate(course_list)}
        self.universe = set(course_list)

        num_users = len(user_list)
        num_items = len(course_list)

        min_rating = train_df["rating"].min()
        max_rating = train_df["rating"].max()

        x = np.column_stack(
            [
                train_df["user"].map(self.user_id2idx).to_numpy(),
                train_df["item"].map(self.course_id2idx).to_numpy(),
            ]
        )
        y = ((train_df["rating"] - min_rating) / (max_rating - min_rating)).to_numpy()

        self.model = RecommenderNet(num_users, num_items, self.embedding_size)
        self.model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )
        self.model.fit(x=x, y=y, batch_size=self.batch_size, epochs=self.epochs, verbose=0)

        self.user_embeddings = self.model.get_layer("user_embedding_layer").get_weights()[0]
        self.course_embeddings = self.model.get_layer("item_embedding_layer").get_weights()[0]
        self.user_bias = self.model.get_layer("user_bias").get_weights()[0].flatten()
        self.course_bias = self.model.get_layer("item_bias").get_weights()[0].flatten()

        self._export_embeddings()
        return self

    def _export_embeddings(self):
        os.makedirs(os.path.dirname(self.user_embeddings_path), exist_ok=True)
        n = self.embedding_size

        user_ids_sorted = [None] * len(self.user_id2idx)
        for uid, idx in self.user_id2idx.items():
            user_ids_sorted[idx] = uid
        user_df = pd.DataFrame(self.user_embeddings, columns=[f"UFeature{i}" for i in range(n)])
        user_df.insert(0, "user", user_ids_sorted)
        user_df.to_csv(self.user_embeddings_path, index=False)
        self.user_embeddings_df = user_df

        course_ids_sorted = [None] * len(self.course_id2idx)
        for cid, idx in self.course_id2idx.items():
            course_ids_sorted[idx] = cid
        course_df = pd.DataFrame(self.course_embeddings, columns=[f"CFeature{i}" for i in range(n)])
        course_df.insert(0, "item", course_ids_sorted)
        course_df.to_csv(self.course_embeddings_path, index=False)
        self.course_embeddings_df = course_df

    def recommend(self, user_id, k: int, exclude: set) -> pd.DataFrame:
        empty = pd.DataFrame(columns=["COURSE_ID", "SCORE"])

        u_idx = self.user_id2idx.get(user_id)
        if u_idx is None:
            return empty

        user_vec = self.user_embeddings[u_idx]
        scores = self.course_embeddings @ user_vec + self.user_bias[u_idx] + self.course_bias
        scores = np.maximum(scores, 0)

        candidates = [
            (course_id, scores[idx])
            for course_id, idx in self.course_id2idx.items()
            if course_id not in exclude
        ]
        if not candidates:
            return empty

        candidates.sort(key=lambda pair: pair[1], reverse=True)
        return pd.DataFrame(candidates[:k], columns=["COURSE_ID", "SCORE"])
