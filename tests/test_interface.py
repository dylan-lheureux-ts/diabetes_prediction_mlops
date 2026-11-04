import pytest

from src.app import (
    parse_structured_input,
    validate_patient_features,
    patient_to_dataframe,
)


@pytest.fixture
def valid_patient():
    return {
        "gender": "Female",
        "age": 45,
        "hypertension": 0,
        "heart_disease": 0,
        "smoking_history": "never",
        "bmi": 27.5,
        "HbA1c_level": 5.8,
        "blood_glucose_level": 120,
    }


def test_parses_valid_patient_input(
    valid_patient,
):
    """
    Valid structured input should be converted into
    correctly typed model features.
    """

    patient = parse_structured_input(
        valid_patient
    )

    assert patient["gender"] == "Female"
    assert patient["age"] == 45.0
    assert patient["hypertension"] == 0
    assert patient["heart_disease"] == 0
    assert patient["smoking_history"] == "never"
    assert patient["bmi"] == 27.5
    assert patient["HbA1c_level"] == 5.8
    assert (
        patient["blood_glucose_level"]
        == 120.0
    )


def test_missing_feature_is_handled_gracefully(
    valid_patient,
):
    """
    Missing required features should produce a clear
    validation error instead of crashing unpredictably.
    """

    incomplete_patient = (
        valid_patient.copy()
    )

    incomplete_patient.pop(
        "HbA1c_level"
    )

    with pytest.raises(
        ValueError,
        match="Missing required features",
    ):
        parse_structured_input(
            incomplete_patient
        )


def test_invalid_binary_feature_raises_error(
    valid_patient,
):
    """
    Binary medical-history fields should only accept
    0 or 1.
    """

    invalid_patient = (
        valid_patient.copy()
    )

    invalid_patient[
        "hypertension"
    ] = 2

    with pytest.raises(
        ValueError,
        match="Hypertension must be 0 or 1",
    ):
        parse_structured_input(
            invalid_patient
        )


def test_patient_converts_to_dataframe(
    valid_patient,
):
    """
    Patient data should be converted into one model-ready row.
    """

    patient = parse_structured_input(
        valid_patient
    )

    df = patient_to_dataframe(
        patient
    )

    assert len(df) == 1
    assert df.shape[1] == 8
    assert "gender" in df.columns
    assert "blood_glucose_level" in df.columns
