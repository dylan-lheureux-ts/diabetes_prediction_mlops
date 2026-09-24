import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    validate_dataframe,
    clean_data,
    encode_categoricals,
    check_data_quality,
)


# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ============================================================
# SAMPLE DATA
# ============================================================

@pytest.fixture
def sample_data():
    """
    Small dataset based on the UCI Heart Disease dataset
    structure used by the project.
    """

    return pd.DataFrame({
        "age": [63.0, 37.0, 41.0, np.nan, 56.0, 57.0],
        "sex": ["Male", "Male", "Female", "Male", "Female", "Female"],
        "cp": [
            "typical angina",
            "asymptomatic",
            "atypical angina",
            "non-anginal",
            "atypical angina",
            "asymptomatic",
        ],
        "trestbps": [145.0, 130.0, 130.0, 120.0, 140.0, 130.0],
        "chol": [233.0, 250.0, 204.0, np.nan, 294.0, 236.0],
        "thalch": [150.0, 187.0, 172.0, 160.0, 153.0, 174.0],
        "oldpeak": [2.3, 3.5, 1.4, 0.0, 1.3, 0.0],
        "num": [0, 1, 0, 1, 0, 0],
    })


# ============================================================
# UNIT TESTS
# ============================================================

class TestValidateDataframe:

    def test_valid_dataframe_passes(self, sample_data):
        """Valid dataframe should return True."""

        result = validate_dataframe(
            sample_data,
            required_columns=["age", "sex", "num"],
            target_column="num",
        )

        assert result is True

    def test_missing_required_column_raises_error(self, sample_data):
        """Missing required columns should raise ValueError."""

        with pytest.raises(
            ValueError,
            match="Missing required columns"
        ):
            validate_dataframe(
                sample_data,
                required_columns=["age", "missing_column"],
                target_column="num",
            )

    def test_missing_target_column_raises_error(self, sample_data):
        """Missing target column should raise ValueError."""

        with pytest.raises(
            ValueError,
            match="Target column"
        ):
            validate_dataframe(
                sample_data,
                required_columns=["age"],
                target_column="missing_target",
            )

    def test_empty_dataframe_raises_error(self):
        """Empty dataframe should raise ValueError."""

        empty_df = pd.DataFrame({
            "age": [],
            "num": [],
        })

        with pytest.raises(
            ValueError,
            match="empty"
        ):
            validate_dataframe(
                empty_df,
                required_columns=["age"],
                target_column="num",
            )


class TestCleanData:

    def test_fills_numeric_missing_values(self, sample_data):
        """Missing numeric values should be filled."""

        result = clean_data(
            sample_data,
            numeric_columns=["age", "chol"],
            categorical_columns=[],
            missing_strategy="median",
        )

        assert result["age"].isna().sum() == 0
        assert result["chol"].isna().sum() == 0

    def test_does_not_modify_original_dataframe(self, sample_data):
        """clean_data should not modify the original dataframe."""

        original = sample_data.copy(deep=True)

        clean_data(
            sample_data,
            numeric_columns=["age", "chol"],
            categorical_columns=[],
            missing_strategy="median",
        )

        pd.testing.assert_frame_equal(sample_data, original)

    def test_fills_numeric_values_with_median(self, sample_data):
        """Numeric missing values should be replaced with the median."""

        result = clean_data(
            sample_data,
            numeric_columns=["age"],
            categorical_columns=[],
            missing_strategy="median",
        )

        # Non-null ages:
        # 63, 37, 41, 56, 57
        # Median = 56
        assert result["age"].iloc[3] == 56.0

    def test_fills_categorical_values_with_mode(self):
        """Missing categorical values should be filled with the mode."""

        data = pd.DataFrame({
            "sex": ["Male", "Female", "Female", None, "Female"],
            "age": [20, 21, 22, 23, 24],
        })

        result = clean_data(
            data,
            numeric_columns=[],
            categorical_columns=["sex"],
            missing_strategy="median",
        )

        assert result["sex"].isna().sum() == 0
        assert result["sex"].iloc[3] == "Female"

    def test_invalid_missing_strategy_raises_error(self, sample_data):
        """Unknown missing-value strategies should raise ValueError."""

        with pytest.raises(
            ValueError,
            match="Unknown missing-value strategy"
        ):
            clean_data(
                sample_data,
                numeric_columns=["age"],
                categorical_columns=[],
                missing_strategy="invalid_strategy",
            )

    def test_drop_strategy_removes_missing_rows(self, sample_data):
        """The drop strategy should remove rows containing missing values."""

        result = clean_data(
            sample_data,
            numeric_columns=["age", "chol"],
            categorical_columns=[],
            missing_strategy="drop",
        )

        assert len(result) == 5
        assert result.isna().sum().sum() == 0


class TestEncodeCategoricals:

    def test_creates_dummy_columns(self, sample_data):
        """Categorical columns should be converted to dummy variables."""

        result = encode_categoricals(
            sample_data,
            columns=["sex"],
        )

        assert "sex" not in result.columns

        gender_columns = [
            column
            for column in result.columns
            if column.startswith("sex_")
        ]

        assert len(gender_columns) > 0

    def test_drops_first_category(self, sample_data):
        """drop_first=True should remove one category."""

        result = encode_categoricals(
            sample_data,
            columns=["sex"],
        )

        sex_columns = [
            column
            for column in result.columns
            if column.startswith("sex_")
        ]

        # sex has two categories, so drop_first=True leaves one column.
        assert len(sex_columns) == 1

    def test_preserves_row_count(self, sample_data):
        """Encoding should not change the number of rows."""

        result = encode_categoricals(
            sample_data,
            columns=["sex", "cp"],
        )

        assert len(result) == len(sample_data)


# ============================================================
# DATA QUALITY TESTS
# ============================================================

class TestDataQuality:

    def test_counts_nulls(self, sample_data):
        """Data quality report should correctly count missing values."""

        report = check_data_quality(
            sample_data,
            numeric_columns=["age", "chol"],
        )

        # One missing age + one missing cholesterol.
        assert report["total_nulls"] == 2

    def test_counts_rows(self, sample_data):
        """Data quality report should correctly count rows."""

        report = check_data_quality(
            sample_data,
            numeric_columns=["age"],
        )

        assert report["total_rows"] == 6

    def test_reports_numeric_ranges(self, sample_data):
        """Data quality report should report min/max values."""

        report = check_data_quality(
            sample_data,
            numeric_columns=["trestbps"],
        )

        assert report["trestbps_min"] == 120.0
        assert report["trestbps_max"] == 145.0


# ============================================================
# ACTUAL DATASET VALIDATION
# ============================================================

@pytest.fixture(scope="module")
def actual_dataset():
    """
    Load the actual Kaggle heart disease dataset.
    """

    path = Path(__file__).resolve(
    ).parents[1] / "data" / "heart_disease_uci.csv"

    assert path.exists(), (
        f"Dataset not found at {path}"
    )

    return pd.read_csv(path)


class TestActualDataset:

    def test_expected_columns_are_present(self, actual_dataset):
        """Actual dataset should contain expected columns."""

        expected_columns = {
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "thalch",
            "oldpeak",
            "num",
        }

        missing_columns = (
            expected_columns - set(actual_dataset.columns)
        )

        assert not missing_columns, (
            f"Missing expected columns: {missing_columns}"
        )

    def test_target_contains_expected_values(self, actual_dataset):
        """
        Original target should contain only values 0-4.

        The experiment later converts these to binary:
        0 = no disease
        1-4 = disease
        """

        target_values = set(
            actual_dataset["num"].dropna().unique()
        )

        assert target_values.issubset({0, 1, 2, 3, 4}), (
            f"Unexpected target values: {target_values}"
        )

    def test_numeric_features_are_within_expected_ranges(
        self,
        actual_dataset,
    ):
        """Numeric heart disease features should be reasonable."""

        age = actual_dataset["age"].dropna()

        assert age.between(18, 100).all()

        trestbps = actual_dataset["trestbps"].dropna()

        # 0 is used as a missing/invalid placeholder in the raw dataset
        trestbps = trestbps[trestbps > 0]

        assert trestbps.between(50, 250).all()

        chol = actual_dataset["chol"].dropna()

        assert chol.between(0, 700).all()

        thalch = actual_dataset["thalch"].dropna()

        assert thalch.between(50, 250).all()


# ============================================================
# MODEL VALIDATION TESTS
# ============================================================

@pytest.fixture(scope="module")
def model_dataset():
    """
    Load and prepare the actual dataset using the same
    preprocessing function used by the experiment.
    """

    from src.experiment import load_and_prepare_data

    config = {
        "features_to_drop": [],
        "target_column": "num",
        "handle_missing": "median",
    }

    X, y, _, _, _ = load_and_prepare_data(config)

    return X, y


@pytest.fixture
def trained_model(model_dataset):
    """Train the configured model on a small sample."""

    from src.experiment import build_model
    from sklearn.model_selection import train_test_split

    X, y = model_dataset

    X = X.iloc[:300]
    y = y.iloc[:300]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    config = {
        "model_type": "gradient_boosting",
        "gb_n_estimators": 50,
        "gb_learning_rate": 0.05,
        "gb_max_depth": 2,
        "random_state": 42,
    }

    model = build_model(config)

    model.fit(X_train, y_train)

    return model, X_test, y_test


def test_model_predictions_have_correct_shape_and_type(
    trained_model,
):
    """Predictions should have the correct type and shape."""

    model, X_test, y_test = trained_model

    predictions = model.predict(X_test)

    assert isinstance(predictions, np.ndarray)

    assert predictions.shape == y_test.shape

    assert len(predictions) == len(X_test)

    assert set(predictions).issubset({0, 1})


def test_model_meets_minimum_accuracy_threshold(
    trained_model,
):
    """Model should achieve at least 70% accuracy."""

    from sklearn.metrics import accuracy_score

    model, X_test, y_test = trained_model

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    assert accuracy >= 0.70, (
        f"Model accuracy {accuracy:.3f} "
        f"is below the required minimum of 0.70"
    )
