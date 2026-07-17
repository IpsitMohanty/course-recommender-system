"""Shared fixtures for the model interface test suite.

Uses a tiny synthetic dataset (12 users, 6 courses) instead of the full
project data so tests run fast and don't depend on network access or the
multi-minute fits some models need on the real ~34k-user dataset. The goal
of this suite isn't accuracy -- it's proving the Recommender interface
contract holds for all 9 models.
"""

import inspect

import pandas as pd
import pytest

RANDOM_STATE = 123


@pytest.fixture
def train_df():
    """12 users, 6 courses. Includes single-enrollment users (5, 8, 11),
    mirroring the real dataset's sparse cohort, and multi-enrollment users
    (1, 7, 12) for the exclude/leak tests."""
    rows = [
        (1, "C1", 5), (1, "C2", 4),
        (2, "C1", 3), (2, "C3", 5),
        (3, "C2", 4), (3, "C4", 3),
        (4, "C1", 5), (4, "C5", 4), (4, "C6", 3),
        (5, "C3", 5),
        (6, "C4", 4), (6, "C5", 5),
        (7, "C1", 3), (7, "C2", 5), (7, "C3", 4),
        (8, "C6", 5),
        (9, "C1", 4), (9, "C4", 5),
        (10, "C2", 3), (10, "C5", 4),
        (11, "C3", 3),
        (12, "C6", 4), (12, "C1", 5), (12, "C2", 3),
    ]
    return pd.DataFrame(rows, columns=["user", "item", "rating"])


@pytest.fixture
def course_genres_df():
    """Tiny synthetic genre catalog matching train_df's 6 courses."""
    rows = [
        ("C1", "Course 1", 1, 0, 1, 0),
        ("C2", "Course 2", 1, 1, 0, 0),
        ("C3", "Course 3", 0, 1, 1, 0),
        ("C4", "Course 4", 0, 0, 1, 1),
        ("C5", "Course 5", 1, 0, 0, 1),
        ("C6", "Course 6", 0, 1, 0, 1),
    ]
    return pd.DataFrame(rows, columns=["COURSE_ID", "TITLE", "GenreA", "GenreB", "GenreC", "GenreD"])


@pytest.fixture
def bows_df():
    """Tiny synthetic BoW features matching train_df's 6 courses."""
    tokens_per_course = {
        "C1": {"python": 2, "data": 1},
        "C2": {"python": 1, "sql": 2},
        "C3": {"data": 2, "science": 1},
        "C4": {"science": 2, "stats": 1},
        "C5": {"python": 1, "stats": 2},
        "C6": {"sql": 1, "data": 1, "science": 1},
    }
    rows = []
    for i, (course_id, tokens) in enumerate(tokens_per_course.items()):
        for token, count in tokens.items():
            rows.append({"doc_index": i, "doc_id": course_id, "token": token, "bow": count})
    return pd.DataFrame(rows)


@pytest.fixture
def isolate_cf_ann_artifacts(monkeypatch, tmp_path):
    """Redirect cf_ann's embedding export/import paths to a temp directory for the
    duration of the test, so fitting cf_ann on the tiny fixture never touches the
    real project's data/user_embeddings_train.csv / data/course_embeddings_train.csv.

    Patched in all three modules that reference the constants: `from ... import X`
    creates a separate local binding in each importing module, so patching
    src.models.cf_ann alone would not affect src.models.cf_classification's or
    src.models.cf_regression's own copy of the name.
    """
    user_path = str(tmp_path / "user_embeddings_train.csv")
    course_path = str(tmp_path / "course_embeddings_train.csv")

    for module_name in ("src.models.cf_ann", "src.models.cf_classification", "src.models.cf_regression"):
        monkeypatch.setattr(f"{module_name}.USER_EMBEDDINGS_PATH", user_path)
        monkeypatch.setattr(f"{module_name}.COURSE_EMBEDDINGS_PATH", course_path)

    return user_path, course_path


# ---------------------------------------------------------------------------
# Model factories: one per model, returning an unfitted instance wired to the
# tiny fixtures. cf_classification/cf_regression each fit a throwaway cf_ann
# first, mirroring the real orchestration dependency (cf_ann must fit and
# export before those two can even be constructed).
# ---------------------------------------------------------------------------

def _make_content_user_profile(course_genres_df, bows_df, train_df):
    from src.models.content_user_profile import ContentUserProfileRecommender
    return ContentUserProfileRecommender(course_genres_df)


def _make_content_course_similarity(course_genres_df, bows_df, train_df):
    from src.models.content_course_similarity import ContentCourseSimilarityRecommender
    return ContentCourseSimilarityRecommender(course_genres_df, bows_df)


def _make_content_clustering(course_genres_df, bows_df, train_df):
    from src.models.content_clustering import ContentClusteringRecommender
    return ContentClusteringRecommender(course_genres_df, max_k=5)


def _make_cf_ann(course_genres_df, bows_df, train_df):
    from src.models.cf_ann import CFAnnRecommender
    return CFAnnRecommender(embedding_size=4, epochs=3, batch_size=8)


def _make_cf_classification(course_genres_df, bows_df, train_df):
    from src.models.cf_ann import CFAnnRecommender
    from src.models.cf_classification import CFClassificationRecommender
    CFAnnRecommender(embedding_size=4, epochs=3, batch_size=8).fit(train_df)
    return CFClassificationRecommender()


def _make_cf_regression(course_genres_df, bows_df, train_df):
    from src.models.cf_ann import CFAnnRecommender
    from src.models.cf_regression import CFRegressionRecommender
    CFAnnRecommender(embedding_size=4, epochs=3, batch_size=8).fit(train_df)
    return CFRegressionRecommender()


def _make_cf_knn(course_genres_df, bows_df, train_df):
    from src.models.cf_knn import CFKnnRecommender
    return CFKnnRecommender()


def _make_cf_nmf(course_genres_df, bows_df, train_df):
    from src.models.cf_nmf import CFNmfRecommender
    return CFNmfRecommender(n_factors=2)


def _make_popularity_baseline(course_genres_df, bows_df, train_df):
    from src.models.popularity_baseline import PopularityBaselineRecommender
    return PopularityBaselineRecommender()


MODEL_FACTORIES = {
    "content_user_profile": _make_content_user_profile,
    "content_course_similarity": _make_content_course_similarity,
    "content_clustering": _make_content_clustering,
    "cf_ann": _make_cf_ann,
    "cf_classification": _make_cf_classification,
    "cf_regression": _make_cf_regression,
    "cf_knn": _make_cf_knn,
    "cf_nmf": _make_cf_nmf,
    "popularity_baseline": _make_popularity_baseline,
}


def make_model(model_name, course_genres_df, bows_df, train_df):
    """Construct (but don't fit) a fresh model instance by name."""
    factory = MODEL_FACTORIES[model_name]
    assert "train_df" in inspect.signature(factory).parameters
    return factory(course_genres_df, bows_df, train_df)


@pytest.fixture(params=sorted(MODEL_FACTORIES.keys()))
def model_name(request):
    return request.param


@pytest.fixture
def unfitted_model(model_name, course_genres_df, bows_df, train_df, isolate_cf_ann_artifacts):
    return make_model(model_name, course_genres_df, bows_df, train_df)


@pytest.fixture
def fitted_model(unfitted_model, train_df):
    return unfitted_model.fit(train_df)
