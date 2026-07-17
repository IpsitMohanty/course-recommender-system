"""Content-based recommender using KMeans clustering on user genre-profile vectors.

Ported from lab_jupyter_content_clustering.ipynb. Builds each user's profile
the same way as content_user_profile (train-only rating x genre matmul),
standardizes, elbow-searches n_clusters, PCA-reduces to >=90% variance,
clusters in PCA space, then recommends the courses most popular (highest
enrollment fraction) within a user's cluster.

Recommendation universe is the courses with actual train-split enrollments
(~126) -- "popularity" is undefined for a course nobody in train has
enrolled in, so it can never be a candidate. Only courses with score > 0
are ever recommended.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.data import load_course_genres
from src.models.base import RANDOM_STATE, Recommender

assert RANDOM_STATE == 123  # keep the global seed visible at the call site


def _find_elbow(k_values, inertia_values):
    """Elbow = point of maximum perpendicular distance from the chord connecting the
    first and last points of the inertia curve. Avoids hardcoding a k."""
    k_arr = np.array(k_values, dtype=float)
    inertia_arr = np.array(inertia_values, dtype=float)
    k_norm = (k_arr - k_arr.min()) / (k_arr.max() - k_arr.min())
    inertia_norm = (inertia_arr - inertia_arr.min()) / (inertia_arr.max() - inertia_arr.min())
    p1 = np.array([k_norm[0], inertia_norm[0]])
    p2 = np.array([k_norm[-1], inertia_norm[-1]])
    line_vec = (p2 - p1) / np.linalg.norm(p2 - p1)
    distances = []
    for i in range(len(k_norm)):
        p = np.array([k_norm[i], inertia_norm[i]])
        proj = p1 + np.dot(p - p1, line_vec) * line_vec
        distances.append(np.linalg.norm(p - proj))
    return k_values[int(np.argmax(distances))]


class ContentClusteringRecommender(Recommender):
    def __init__(self, course_genres_df: pd.DataFrame = None, max_k: int = 30, variance_threshold: float = 0.9):
        self.course_genres_df = course_genres_df if course_genres_df is not None else load_course_genres()
        self.course_order = self.course_genres_df["COURSE_ID"].tolist()
        self.genre_cols = list(self.course_genres_df.columns[2:])
        self.genre_matrix = self.course_genres_df[self.genre_cols].to_numpy()
        self.max_k = max_k
        self.variance_threshold = variance_threshold

        self.chosen_k = None
        self.chosen_n_components = None
        self._cluster_by_user = {}
        self._popularity_by_cluster = {}  # cluster_id -> {course_id: fraction}

    def fit(self, train_df: pd.DataFrame) -> "ContentClusteringRecommender":
        rating_matrix = (
            train_df.pivot_table(index="user", columns="item", values="rating", aggfunc="max")
            .reindex(columns=self.course_order, fill_value=0)
            .fillna(0)
        )
        profile_matrix = rating_matrix.to_numpy() @ self.genre_matrix
        user_ids = rating_matrix.index.tolist()

        scaler = StandardScaler()
        features = scaler.fit_transform(profile_matrix)

        list_k = list(range(1, min(self.max_k, len(features)) + 1))
        inertias = []
        for k in list_k:
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(features)
            inertias.append(km.inertia_)
        self.chosen_k = _find_elbow(list_k, inertias)

        n_components_range = list(range(1, len(self.genre_cols) + 1))
        variance_ratios = []
        for n in n_components_range:
            pca_test = PCA(n_components=n, random_state=RANDOM_STATE)
            pca_test.fit(features)
            variance_ratios.append(pca_test.explained_variance_ratio_.sum())
        self.chosen_n_components = next(
            n for n, v in zip(n_components_range, variance_ratios) if v >= self.variance_threshold
        )

        pca = PCA(n_components=self.chosen_n_components, random_state=RANDOM_STATE)
        components = pca.fit_transform(features)

        kmeans = KMeans(n_clusters=self.chosen_k, random_state=RANDOM_STATE, n_init=10).fit(components)
        self._cluster_by_user = dict(zip(user_ids, kmeans.labels_))

        cluster_of = pd.Series(self._cluster_by_user, name="cluster").rename_axis("user").reset_index()
        labelled = train_df.merge(cluster_of, on="user", how="inner")
        cluster_sizes = labelled.groupby("cluster")["user"].nunique()
        counts = labelled.groupby(["cluster", "item"]).size().rename("enrollments").reset_index()
        pivot_pop = counts.pivot(index="cluster", columns="item", values="enrollments").fillna(0)
        pop_fraction = pivot_pop.div(cluster_sizes, axis=0)

        self._popularity_by_cluster = {
            cluster_id: pop_fraction.loc[cluster_id].to_dict() for cluster_id in pop_fraction.index
        }
        self.universe = {
            course_id
            for popularity in self._popularity_by_cluster.values()
            for course_id, score in popularity.items()
            if score > 0
        }
        return self

    def recommend(self, user_id, k: int, exclude: set) -> pd.DataFrame:
        empty = pd.DataFrame(columns=["COURSE_ID", "SCORE"])

        cluster_id = self._cluster_by_user.get(user_id)
        if cluster_id is None:
            return empty
        popularity = self._popularity_by_cluster.get(cluster_id, {})
        if not popularity:
            return empty

        candidates = [
            (course_id, score)
            for course_id, score in popularity.items()
            if course_id not in exclude and score > 0
        ]
        if not candidates:
            return empty

        candidates.sort(key=lambda pair: pair[1], reverse=True)
        return pd.DataFrame(candidates[:k], columns=["COURSE_ID", "SCORE"])
