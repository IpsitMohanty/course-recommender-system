"""Interface-contract tests for every model in src/models/.

Not exhaustive coverage -- the handful of tests that prove the
Recommender contract (src/models/base.py) holds for all 9 models, plus
regression tests for two bugs this project actually hit:

- cf_nmf originally had no `rs` defined at all (a NameError caught only at
  notebook-execution time) -- test_determinism guards the whole class of
  "did we actually seed this" bugs, not just that one.
- Every prediction file before this package existed masked a model's own
  full rating history internally, making it structurally impossible to
  ever recommend a held-out test course -- test_no_full_history_leak
  guards against that design reappearing.
"""

import pandas as pd
import pytest

from tests.conftest import RANDOM_STATE, make_model

ALL_COURSES = {"C1", "C2", "C3", "C4", "C5", "C6"}
K = 6  # >= universe size, so truncation never hides a masking bug


def test_fit_does_not_raise(unfitted_model, train_df):
    result = unfitted_model.fit(train_df)
    assert result is unfitted_model  # fit() returns self per the interface contract


def test_recommend_output_schema(fitted_model, train_df):
    recs = fitted_model.recommend(1, K, exclude={"C1"})

    assert isinstance(recs, pd.DataFrame)
    assert list(recs.columns) == ["COURSE_ID", "SCORE"]
    assert len(recs) <= K
    if not recs.empty:
        assert pd.api.types.is_numeric_dtype(recs["SCORE"])
        assert set(recs["COURSE_ID"]).issubset(ALL_COURSES)


def test_recommend_respects_exclude(fitted_model, train_df):
    """The non-negotiable one: nothing in `exclude` may ever appear in the output.
    This property is what the entire evaluation harness's correctness rests on."""
    exclude = {"C1", "C2"}
    recs = fitted_model.recommend(1, K, exclude=exclude)

    assert exclude.isdisjoint(set(recs["COURSE_ID"]))


def test_recommend_ranking_order(fitted_model, train_df):
    recs = fitted_model.recommend(1, K, exclude={"C1"})
    if len(recs) > 1:
        assert recs["SCORE"].is_monotonic_decreasing


def test_recommend_unknown_user_does_not_crash(fitted_model, train_df):
    """A user absent from train_df entirely must not crash recommend() --
    it should return a valid (possibly empty) DataFrame with the right schema."""
    recs = fitted_model.recommend(999999, K, exclude=set())

    assert isinstance(recs, pd.DataFrame)
    assert list(recs.columns) == ["COURSE_ID", "SCORE"]


def test_determinism(model_name, course_genres_df, bows_df, train_df, isolate_cf_ann_artifacts):
    """Fitting the same model twice with rs=123 on identical data must give identical
    recommendations. Regression test for the class of bug cf_nmf hit: a model that
    forgets to seed its own randomness (or never defines rs at all)."""
    model_a = make_model(model_name, course_genres_df, bows_df, train_df).fit(train_df)
    model_b = make_model(model_name, course_genres_df, bows_df, train_df).fit(train_df)

    recs_a = model_a.recommend(1, K, exclude={"C1"})
    recs_b = model_b.recommend(1, K, exclude={"C1"})

    assert recs_a["COURSE_ID"].tolist() == recs_b["COURSE_ID"].tolist()
    if not recs_a.empty:
        assert recs_a["SCORE"].tolist() == pytest.approx(recs_b["SCORE"].tolist())


def test_no_full_history_leak(fitted_model, train_df):
    """User 7 knows {C1, C2, C3}. Excluding only C1 must leave a strictly larger
    eligible-candidate pool than excluding all three -- if a model still masked
    against the user's full history internally (the old prediction-file bug),
    varying `exclude` would have no effect and the two pools would be identical."""
    full_history = {"C1", "C2", "C3"}
    partial_exclude = {"C1"}

    recs_partial = fitted_model.recommend(7, K, exclude=partial_exclude)
    recs_full = fitted_model.recommend(7, K, exclude=full_history)

    eligible_partial = set(recs_partial["COURSE_ID"])
    eligible_full = set(recs_full["COURSE_ID"])

    assert partial_exclude.isdisjoint(eligible_partial)
    assert full_history.isdisjoint(eligible_full)

    # Courses masked only in the "full" case must remain eligible in the "partial" case.
    assert eligible_full <= eligible_partial
    assert len(eligible_partial - eligible_full) > 0
