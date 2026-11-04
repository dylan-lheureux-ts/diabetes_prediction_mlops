import pytest

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.train import (
    config,
    load_data,
    build_preprocessor,
    build_model,
)


@pytest.fixture(scope="module")
def trained_model_data():
    """
    Train a small model using the diabetes training pipeline
    for model validation tests.
    """

    test_config = config.copy()

    # Logistic Regression is fast and deterministic for tests
    test_config["model_type"] = "logistic_regression"
    test_config["lr_C"] = 1.0

    X, y, _ = load_data(test_config)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_config["test_size"],
        random_state=test_config["random_state"],
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(test_config),
            ),
            (
                "classifier",
                build_model(test_config),
            ),
        ]
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    return (
        pipeline,
        X_test,
        y_test,
    )


def test_model_predictions_have_correct_shape_and_type(
    trained_model_data,
):
    """
    Model predictions should match the number of test rows
    and contain valid binary class labels.
    """

    model, X_test, _ = trained_model_data

    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)

    assert set(predictions).issubset(
        {0, 1}
    )


def test_model_meets_minimum_accuracy_threshold(
    trained_model_data,
):
    """
    Model should meet the minimum expected accuracy
    on the held-out test set.
    """

    model, X_test, y_test = trained_model_data

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    assert accuracy >= 0.70
