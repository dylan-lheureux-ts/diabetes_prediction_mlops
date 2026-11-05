from unittest.mock import MagicMock

import pytest
import src.app as app

from src.app import (
    parse_structured_input,
    validate_patient_features,
    patient_to_dataframe,
    format_prediction_response,
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


def test_parse_natural_language_input(monkeypatch):
    """
    Test that natural-language input is converted
    into the expected validated patient dictionary
    without making a real OpenRouter API request.
    """

    monkeypatch.setattr(
        app,
        "OPENROUTER_API_KEY",
        "test-api-key",
    )

    monkeypatch.setattr(
        app,
        "OPENROUTER_MODEL",
        "test-model",
    )

    mock_response = MagicMock()

    mock_response.choices[
        0
    ].message.content = """
    {
        "gender": "Male",
        "age": 58,
        "hypertension": 1,
        "heart_disease": 0,
        "smoking_history": "former",
        "bmi": 30.5,
        "HbA1c_level": 6.7,
        "blood_glucose_level": 160
    }
    """

    mock_client = MagicMock()

    mock_client.chat.completions.create.return_value = (
        mock_response
    )

    monkeypatch.setattr(
        app,
        "OpenAI",
        lambda **kwargs: mock_client,
    )

    patient = app.parse_natural_language_input(
        """
        I am a 58-year-old male.
        I have hypertension but no heart disease.
        I used to smoke.
        My BMI is 30.5.
        My HbA1c is 6.7.
        My blood glucose is 160.
        """
    )

    assert patient["gender"] == "Male"
    assert patient["age"] == 58.0
    assert patient["hypertension"] == 1
    assert patient["heart_disease"] == 0
    assert patient["smoking_history"] == "former"
    assert patient["bmi"] == 30.5
    assert patient["HbA1c_level"] == 6.7
    assert patient["blood_glucose_level"] == 160.0


def test_natural_language_missing_features(
    monkeypatch,
):
    """
    Test graceful handling when the user's description
    does not contain all required model features.
    """

    monkeypatch.setattr(
        app,
        "OPENROUTER_API_KEY",
        "test-api-key",
    )

    monkeypatch.setattr(
        app,
        "OPENROUTER_MODEL",
        "test-model",
    )

    mock_response = MagicMock()

    mock_response.choices[
        0
    ].message.content = """
    {
        "gender": null,
        "age": 55,
        "hypertension": null,
        "heart_disease": null,
        "smoking_history": null,
        "bmi": null,
        "HbA1c_level": null,
        "blood_glucose_level": 150
    }
    """

    mock_client = MagicMock()

    mock_client.chat.completions.create.return_value = (
        mock_response
    )

    monkeypatch.setattr(
        app,
        "OpenAI",
        lambda **kwargs: mock_client,
    )

    with pytest.raises(
        ValueError,
        match="Missing required features",
    ):
        app.parse_natural_language_input(
            """
            I am 55 years old and my
            blood glucose is 150.
            """
        )


def test_llm_invalid_json(
    monkeypatch,
):
    """
    Test graceful handling when the LLM does not
    return valid JSON.
    """

    monkeypatch.setattr(
        app,
        "OPENROUTER_API_KEY",
        "test-api-key",
    )

    monkeypatch.setattr(
        app,
        "OPENROUTER_MODEL",
        "test-model",
    )

    mock_response = MagicMock()

    mock_response.choices[
        0
    ].message.content = (
        "This is not valid JSON."
    )

    mock_client = MagicMock()

    mock_client.chat.completions.create.return_value = (
        mock_response
    )

    monkeypatch.setattr(
        app,
        "OpenAI",
        lambda **kwargs: mock_client,
    )

    with pytest.raises(
        ValueError,
        match="The LLM returned invalid JSON",
    ):
        app.parse_natural_language_input(
            "Patient information here."
        )

