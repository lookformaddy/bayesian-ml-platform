import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from bayesian.config import Settings
from bayesian.data import generate_dataset, train_test_split
from bayesian.models import BayesianLogisticRegression, PointLogisticRegression
from bayesian.platform import BayesianPlatform
from bayesian.registry import ModelRegistry
from bayesian.serving import BayesianServer


@pytest.fixture()
def settings():
    return Settings()


@pytest.fixture()
def dataset():
    return generate_dataset(n=600, seed=42)


@pytest.fixture()
def split(dataset):
    return train_test_split(dataset, test_fraction=0.25, seed=42)


@pytest.fixture()
def model(split):
    return BayesianLogisticRegression(seed=42).fit(split.X_train, split.y_train)


@pytest.fixture()
def point_model(split):
    return PointLogisticRegression().fit(split.X_train, split.y_train)


@pytest.fixture()
def server(model, point_model, settings):
    return BayesianServer(model, point_model, settings)


@pytest.fixture()
def registry(settings):
    return ModelRegistry(settings)


@pytest.fixture()
def platform(settings):
    return BayesianPlatform(settings)
