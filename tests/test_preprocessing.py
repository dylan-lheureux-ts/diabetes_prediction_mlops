from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    validate_dataframe,
    clean_data,
    encode_categoricals,
    scale_numeric_features,
    check_data_quality,
)


# ============================================================
# SAMPLE DATA
# ============================================================

@pytest.fixture
def sample_data():
    """
    Small synthetic dataset based on the diabetes prediction
    dataset structure used by the project.
    """

    return pd.DataFrame({
        "gender": [
            "Female",
            "Male",
            "Female",
            "Male",
            "Female",
            "Male",
        ],
        "age": [
            32.0,
            45.0,
            np.nan,
            61.0,
            27.0,
            54.0,
        ],
        "hypertension": [
            0,
            1,
            0,
            1,
            0,
            0,
        ],
        "heart_disease": [
            0,
            0,
            0,
            1,
            0,
            1,
        ],
        "smoking_history": [
            "never",
            "former",
            "current",
            None,
            "never",
            "former",
        ],
        "bmi": [
            24.5,
            31.2,
            27.8,
            np.nan,
            22.1,
            29.4,
        ],
        "HbA1c_level": [
            5.1,
            6.8,
            5.7,
            7.2,
            4.9,
            6.1,
        ],
        "blood_glucose_level": [
            95,
            180,
            120,
            210,
            90,
            145,
        ],
        "diabetes": [
            0,
            1,
            0,
            1,
            0,
            0,
        ],
    })

# ============================================================
# UNIT TESTS
# ============================================================


class TestValidateDataframe:

    def test_valid_dataframe_passes(self, sample_data):
        """Valid dataframe should return True."""

        result = validate_dataframe(
            sample_data,
            required_columns=[
                "age",
                "gender",
                "diabetes",
            ],
            target_column="diabetes",
        )

        assert result is True

    def test_missing_required_column_raises_error(
        self,
        sample_data,
    ):
        with pytest.raises(
            ValueError,
            match="Missing required columns",
        ):
            validate_dataframe(
                sample_data,
                required_columns=[
                    "age",
                    "missing_column",
                ],
                target_column="diabetes",
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
            "diabetes": [],
        })

        with pytest.raises(
            ValueError,
            match="empty"
        ):
            validate_dataframe(
                empty_df,
                required_columns=["age"],
                target_column="diabetes",
            )


class TestCleanData:

    def test_fills_numeric_missing_values(self, sample_data):
        """Missing numeric values should be filled."""

        result = clean_data(
            sample_data,
            numeric_columns=["age", "bmi"],
            categorical_columns=[],
            missing_strategy="median",
        )

        assert result["age"].isna().sum() == 0
        assert result["bmi"].isna().sum() == 0

    def test_does_not_modify_original_dataframe(self, sample_data):
        """clean_data should not modify the original dataframe."""

        original = sample_data.copy(deep=True)

        clean_data(
            sample_data,
            numeric_columns=["age", "bmi"],
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
        # 27, 32, 45, 54, 61
        # Median = 45
        assert result["age"].iloc[2] == 45.0

    def test_fills_categorical_values_with_mode(self):
        """Missing categorical values should be filled with the mode."""

        data = pd.DataFrame({
            "gender": ["Male", "Female", "Female", None, "Female"],
            "age": [20, 21, 22, 23, 24],
        })

        result = clean_data(
            data,
            numeric_columns=[],
            categorical_columns=["gender"],
            missing_strategy="median",
        )

        assert result["gender"].isna().sum() == 0
        assert result["gender"].iloc[3] == "Female"

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
            numeric_columns=[
                "age",
                "bmi",
                "HbA1c_level",
                "blood_glucose_level"
            ],
            categorical_columns=["smoking_history", "gender"],
            missing_strategy="drop",
        )

        assert len(result) == 4
        assert result.isna().sum().sum() == 0


class TestEncodeCategoricals:

    def test_creates_dummy_columns(self, sample_data):
        result = encode_categoricals(
            sample_data,
            columns=["gender", "smoking_history"],
        )

        assert "gender" not in result.columns
        assert "smoking_history" not in result.columns

        gender_columns = [
            column
            for column in result.columns
            if column.startswith("gender_")
        ]

        smoking_columns = [
            column
            for column in result.columns
            if column.startswith("smoking_history_")
        ]

        assert len(gender_columns) > 0
        assert len(smoking_columns) > 0

    def test_drops_first_category(self, sample_data):
        result = encode_categoricals(
            sample_data,
            columns=["gender"],
        )

        gender_columns = [
            column
            for column in result.columns
            if column.startswith("gender_")
        ]

        assert len(gender_columns) == 1

    def test_preserves_row_count(self, sample_data):
        result = encode_categoricals(
            sample_data,
            columns=["gender", "smoking_history"],
        )

        assert len(result) == len(sample_data)

# ============================================================
# DATA QUALITY TESTS
# ============================================================


class TestDataQuality:

    def test_counts_nulls(
        self,
        sample_data,
    ):
        report = check_data_quality(
            sample_data,
            numeric_columns=[
                "age",
                "bmi",
            ],
        )

        # Missing age + missing BMI +
        # missing smoking history
        assert report["total_nulls"] == 3

    def test_counts_rows(
        self,
        sample_data,
    ):
        report = check_data_quality(
            sample_data,
            numeric_columns=["age"],
        )

        assert report["total_rows"] == 6

    def test_reports_numeric_ranges(
        self,
        sample_data,
    ):
        report = check_data_quality(
            sample_data,
            numeric_columns=[
                "blood_glucose_level",
            ],
        )

        assert (
            report["blood_glucose_level_min"]
            == 90.0
        )

        assert (
            report["blood_glucose_level_max"]
            == 210.0
        )


# ============================================================
# ACTUAL DATASET VALIDATION
# ============================================================

@pytest.fixture(scope="module")
def actual_dataset():
    """
    Load the actual diabetes prediction dataset.
    """

    path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "diabetes_prediction_dataset.csv"
    )

    assert path.exists(), (
        f"Dataset not found at {path}"
    )

    return pd.read_csv(path)


class TestActualDataset:

    def test_expected_columns_are_present(
        self,
        actual_dataset,
    ):
        expected_columns = {
            "gender",
            "age",
            "hypertension",
            "heart_disease",
            "smoking_history",
            "bmi",
            "HbA1c_level",
            "blood_glucose_level",
            "diabetes",
        }

        assert expected_columns.issubset(
            actual_dataset.columns
        )

    def test_target_contains_expected_values(
        self,
        actual_dataset,
    ):
        target_values = set(
            actual_dataset["diabetes"].unique()
        )

        assert target_values.issubset({0, 1})

    def test_numeric_features_are_within_expected_ranges(
        self,
        actual_dataset,
    ):
        assert (
            actual_dataset["age"] >= 0
        ).all()

        assert (
            actual_dataset["bmi"] > 0
        ).all()

        assert (
            actual_dataset["HbA1c_level"] > 0
        ).all()

        assert (
            actual_dataset["blood_glucose_level"] > 0
        ).all()


# ============================================================
# TEST SCALING OF NUMERIC FEATURES
# ============================================================
class TestScaleNumericFeatures:

    def test_scales_numeric_features(
        self,
        sample_data,
    ):
        clean_df = clean_data(
            sample_data,
            numeric_columns=[
                "age",
                "bmi",
                "HbA1c_level",
                "blood_glucose_level",
            ],
            categorical_columns=[
                "gender",
                "smoking_history",
            ],
            missing_strategy="median",
        )

        scaled_df, scaler = scale_numeric_features(
            clean_df,
            numeric_columns=[
                "age",
                "bmi",
                "HbA1c_level",
                "blood_glucose_level",
            ],
        )

        assert abs(
            scaled_df["age"].mean()
        ) < 1e-10

        assert abs(
            scaled_df["bmi"].mean()
        ) < 1e-10

        assert scaler is not None

    def test_scaling_does_not_modify_original_dataframe(
        self,
        sample_data,
    ):

        clean_df = clean_data(
            sample_data,
            numeric_columns=[
                "age",
                "bmi",
                "HbA1c_level",
                "blood_glucose_level",
            ],
            categorical_columns=[
                "gender",
                "smoking_history",
            ],
            missing_strategy="median",
        )

        original = clean_df.copy(deep=True)

        scale_numeric_features(
            clean_df,
            numeric_columns=[
                "age",
                "bmi",
                "HbA1c_level",
                "blood_glucose_level",
            ],
        )

        pd.testing.assert_frame_equal(
            clean_df,
            original,
        )
