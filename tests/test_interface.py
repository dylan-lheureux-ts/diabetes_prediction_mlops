from unittest.mock import MagicMock

import pytest
import src.app as app

from src.app import (
    answer_prediction_question,
    explain_prediction,
    parse_structured_input,
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


def test_missing_feature_identifies_field(
    valid_patient,
):
    incomplete_patient = (
        valid_patient.copy()
    )

    incomplete_patient.pop(
        "HbA1c_level"
    )

    with pytest.raises(
        ValueError,
        match="HbA1c_level: missing required value",
    ):
        parse_structured_input(
            incomplete_patient
        )


def test_multiple_invalid_features_are_reported(
    valid_patient,
):
    invalid_patient = (
        valid_patient.copy()
    )

    invalid_patient.update(
        {
            "age": -5,
            "hypertension": 2,
            "bmi": -10,
            "HbA1c_level": 0,
            "blood_glucose_level": -20,
        }
    )

    with pytest.raises(
        ValueError
    ) as exc_info:
        parse_structured_input(
            invalid_patient
        )

    message = str(
        exc_info.value
    )

    assert "age: cannot be negative" in message
    assert "hypertension: must be 0 or 1" in message
    assert "bmi: must be greater than zero" in message
    assert "HbA1c_level: must be greater than zero" in message
    assert (
        "blood_glucose_level: must be greater than zero"
        in message
    )


def test_patient_converts_to_dataframe(
    valid_patient,
):
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


def _mock_openrouter(
    monkeypatch,
    response_content,
):
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
    ].message.content = response_content

    mock_client = MagicMock()

    mock_client.chat.completions.create.return_value = (
        mock_response
    )

    monkeypatch.setattr(
        app,
        "OpenAI",
        lambda **kwargs: mock_client,
    )

    return mock_client


def test_parse_natural_language_valid(
    monkeypatch,
):
    _mock_openrouter(
        monkeypatch,
        """
        {
            "status": "valid",
            "patient": {
                "gender": "Male",
                "age": 58,
                "hypertension": 1,
                "heart_disease": 0,
                "smoking_history": "former",
                "bmi": 30.5,
                "HbA1c_level": 6.7,
                "blood_glucose_level": 160
            },
            "issues": [],
            "message": ""
        }
        """,
    )

    parsed = app.parse_natural_language_input(
        """
        I am a 58-year-old male.
        I have hypertension but no heart disease.
        I used to smoke.
        My BMI is 30.5.
        My HbA1c is 6.7.
        My blood glucose is 160.
        """
    )

    assert parsed["status"] == "valid"

    patient = parsed["patient"]

    assert patient["gender"] == "Male"
    assert patient["age"] == 58.0
    assert patient["hypertension"] == 1
    assert patient["heart_disease"] == 0
    assert patient["smoking_history"] == "former"
    assert patient["bmi"] == 30.5
    assert patient["HbA1c_level"] == 6.7
    assert patient["blood_glucose_level"] == 160.0


def test_natural_language_contradiction_identifies_field(
    monkeypatch,
):
    _mock_openrouter(
        monkeypatch,
        """
        {
            "status": "needs_clarification",
            "patient": {
                "gender": "Male",
                "age": 50,
                "hypertension": null,
                "heart_disease": 0,
                "smoking_history": "never",
                "bmi": 28,
                "HbA1c_level": 6.0,
                "blood_glucose_level": 130
            },
            "issues": [
                {
                    "field": "hypertension",
                    "message": "The information is contradictory: you said both that you have hypertension and that you do not have hypertension."
                }
            ],
            "message": "I found conflicting information for hypertension. Please clarify whether you have hypertension."
        }
        """,
    )

    parsed = app.parse_natural_language_input(
        """
        I have hypertension, but I also do not
        have hypertension.
        """
    )

    assert (
        parsed["status"]
        == "needs_clarification"
    )

    assert (
        parsed["issues"][0]["field"]
        == "hypertension"
    )

    assert "contradictory" in (
        parsed["issues"][0]["message"]
    )


def test_natural_language_reports_multiple_issues(
    monkeypatch,
):
    _mock_openrouter(
        monkeypatch,
        """
        {
            "status": "needs_clarification",
            "patient": {
                "gender": null,
                "age": 55,
                "hypertension": null,
                "heart_disease": null,
                "smoking_history": null,
                "bmi": null,
                "HbA1c_level": null,
                "blood_glucose_level": 150
            },
            "issues": [
                {
                    "field": "gender",
                    "message": "Gender is missing."
                },
                {
                    "field": "hypertension",
                    "message": "Hypertension status is missing."
                },
                {
                    "field": "bmi",
                    "message": "BMI is missing."
                }
            ],
            "message": "I need a few more details before I can make a prediction."
        }
        """,
    )

    parsed = app.parse_natural_language_input(
        """
        I am 55 years old and my
        blood glucose is 150.
        """
    )

    assert (
        parsed["status"]
        == "needs_clarification"
    )

    fields = {
        issue["field"]
        for issue in parsed["issues"]
    }

    assert {
        "gender",
        "hypertension",
        "bmi",
    }.issubset(fields)


def test_out_of_scope_request_is_conversational(
    monkeypatch,
):
    _mock_openrouter(
        monkeypatch,
        """
        {
            "status": "out_of_scope",
            "patient": null,
            "issues": [],
            "message": "I can explain a diabetes model prediction, but I cannot recommend medications or provide medical treatment advice."
        }
        """,
    )

    parsed = app.parse_natural_language_input(
        "What medication should I take?"
    )

    assert (
        parsed["status"]
        == "out_of_scope"
    )

    assert parsed["patient"] is None

    assert "cannot recommend medications" in (
        parsed["message"]
    )


def test_llm_invalid_json(
    monkeypatch,
):
    _mock_openrouter(
        monkeypatch,
        "This is not valid JSON.",
    )

    with pytest.raises(
        ValueError,
        match="The LLM returned invalid JSON",
    ):
        app.parse_natural_language_input(
            "Patient information here."
        )


def test_local_feature_contributions_are_ranked(
    monkeypatch,
    valid_patient,
):
    class FakePreprocessor:
        def transform(self, dataframe):
            return [[0] * 8]

        def get_feature_names_out(self):
            return [
                "numeric__age",
                "numeric__bmi",
                "numeric__HbA1c_level",
                "numeric__blood_glucose_level",
                "categorical__gender_Female",
                "categorical__smoking_history_never",
                "binary__hypertension",
                "binary__heart_disease",
            ]

    class FakeBooster:
        def predict(
            self,
            matrix,
            pred_contribs=False,
        ):
            assert pred_contribs is True

            return [[
                0.1,
                0.2,
                0.9,
                0.4,
                -0.1,
                0.05,
                0.02,
                0.01,
                -1.0,
            ]]

    class FakeClassifier:
        def get_booster(self):
            return FakeBooster()

    class FakePipeline:
        named_steps = {
            "preprocessor": FakePreprocessor(),
            "classifier": FakeClassifier(),
        }

    monkeypatch.setattr(
        app.xgb,
        "DMatrix",
        lambda value: value,
    )

    contributions = explain_prediction(
        valid_patient,
        model=FakePipeline(),
    )

    assert (
        contributions[0]["feature"]
        == "HbA1c_level"
    )

    assert (
        contributions[0]["direction"]
        == "increased"
    )

    assert (
        contributions[0][
            "relative_influence_percent"
        ]
        > contributions[1][
            "relative_influence_percent"
        ]
    )


def test_followup_question_uses_grounded_context(
    monkeypatch,
    valid_patient,
):
    mock_client = _mock_openrouter(
        monkeypatch,
        (
            "HbA1c level had the largest influence "
            "on this model prediction."
        ),
    )

    result = {
        "prediction": 1,
        "probability": 0.72,
        "threshold": 0.25,
    }

    contributions = [
        {
            "feature": "HbA1c_level",
            "value": 5.8,
            "direction": "increased",
            "contribution": 0.9,
            "relative_influence_percent": 60.0,
        }
    ]

    answer = answer_prediction_question(
        "Which feature mattered most?",
        valid_patient,
        result,
        contributions,
    )

    assert "HbA1c level" in answer

    mock_client.chat.completions.create.assert_called_once()
