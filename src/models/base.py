"""Shared interface every recommender model implements.

The key design decision: masking is the caller's responsibility, not the
model's. fit() only ever sees train_df, and recommend() only ever masks the
course ids explicitly passed in `exclude`. Earlier prediction files masked
each model's own full rating history internally, which meant no model could
ever surface a held-out test course -- making offline evaluation against a
holdout impossible by construction. Putting `exclude` on the caller fixes
that: evaluation code passes the user's *train* enrollments as `exclude`,
leaving held-out test courses eligible to be recommended and scored.
"""

from abc import ABC, abstractmethod

import pandas as pd

RANDOM_STATE = 123


class Recommender(ABC):
    """Abstract base class for all recommender models in this package."""

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> "Recommender":
        """Learn from train-only interaction data.

        Parameters
        ----------
        train_df : DataFrame with at least columns [user, item, rating].
            Must contain only training-split rows -- never the full history.

        Returns
        -------
        self, so calls can be chained (model = SomeRecommender().fit(train_df)).
        """
        raise NotImplementedError

    @abstractmethod
    def recommend(self, user_id, k: int, exclude: set) -> pd.DataFrame:
        """Return up to k ranked recommendations for a single user.

        Parameters
        ----------
        user_id : the raw user id to recommend for.
        k : maximum number of recommendations to return.
        exclude : set of raw course ids to mask out of the candidates. The
            caller decides what this is -- typically the user's train-split
            enrollments during evaluation, so held-out test courses stay
            eligible to be recommended.

        Returns
        -------
        DataFrame with columns [COURSE_ID, SCORE], sorted by SCORE
        descending, at most k rows. Empty DataFrame (same columns, zero
        rows) if the user is unknown to the fitted model or has no eligible
        candidates.
        """
        raise NotImplementedError
