import json
import cloudpickle
from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_model.pkl"
)

SELECTION_PATH = (
    PROJECT_ROOT
    / "metrics"
    / "best_model_selection.json"
)

REQUIRED_FEATURES = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]

VALID_GENDERS = {
    "Female",
    "Male",
    "Other",
}

VALID_SMOKING_HISTORY = {
    "never",
    "No Info",
    "current",
    "former",
    "ever",
    "not current",
}

load_dotenv()

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY"
)

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL"
)

OPENROUTER_BASE_URL = (
    "https://openrouter.ai/api/v1"
)


def parse_structured_input(data):
    """
    Validate structured patient information.

    All detected problems are collected so the user can
    correct every invalid field at the same time.
    """

    if not isinstance(data, dict):
        raise ValueError(
            "Parsed patient information must be a dictionary."
        )

    issues = []

    # ========================================================
    # Missing values
    # ========================================================

    for feature in REQUIRED_FEATURES:

        if (
            feature not in data
            or data[feature] is None
        ):
            issues.append(
                f"{feature}: missing required value"
            )

    # Stop conversion only after identifying all missing fields
    if issues:
        raise ValueError(
            "Please correct the following patient information: "
            + "; ".join(issues)
        )

    patient = {}

    # ========================================================
    # Numeric conversions
    # ========================================================

    numeric_fields = [
        "age",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level",
    ]

    for field in numeric_fields:

        try:
            patient[field] = float(
                data[field]
            )

        except (TypeError, ValueError):

            issues.append(
                f"{field}: must be a numeric value"
            )

    # ========================================================
    # Binary conversions
    # ========================================================

    binary_fields = [
        "hypertension",
        "heart_disease",
    ]

    for field in binary_fields:

        try:
            patient[field] = int(
                data[field]
            )

        except (TypeError, ValueError):

            issues.append(
                f"{field}: must be 0 or 1"
            )

    # ========================================================
    # Categorical fields
    # ========================================================

    patient["gender"] = str(
        data["gender"]
    )

    patient["smoking_history"] = str(
        data["smoking_history"]
    )

    # ========================================================
    # Validate all converted values
    # ========================================================

    if patient["gender"] not in VALID_GENDERS:

        issues.append(
            "gender: must be Female, Male, or Other"
        )

    if (
        patient["smoking_history"]
        not in VALID_SMOKING_HISTORY
    ):

        issues.append(
            "smoking_history: invalid smoking history"
        )

    if "age" in patient:

        if patient["age"] < 0:

            issues.append(
                "age: cannot be negative"
            )

    if "bmi" in patient:

        if patient["bmi"] <= 0:

            issues.append(
                "bmi: must be greater than zero"
            )

    if "HbA1c_level" in patient:

        if patient["HbA1c_level"] <= 0:

            issues.append(
                "HbA1c_level: must be greater than zero"
            )

    if "blood_glucose_level" in patient:

        if patient["blood_glucose_level"] <= 0:

            issues.append(
                "blood_glucose_level: must be greater than zero"
            )

    if "hypertension" in patient:

        if patient["hypertension"] not in {0, 1}:

            issues.append(
                "hypertension: must be 0 or 1"
            )


    if "heart_disease" in patient:

        if patient["heart_disease"] not in {0, 1}:

            issues.append(
                "heart_disease: must be 0 or 1"
            )

    # ========================================================
    # Return every problem together
    # ========================================================

    if issues:

        raise ValueError(
            "Please correct the following patient information: "
            + "; ".join(issues)
        )

    return patient


def parse_natural_language_input(text):
    """
    Convert a natural-language patient description
    into the validated feature dictionary required
    by the diabetes prediction model.
    """

    if not isinstance(text, str):
        raise ValueError(
            "Patient description must be text."
        )

    if not text.strip():
        raise ValueError(
            "Patient description cannot be empty."
        )

    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY is not configured."
        )

    if not OPENROUTER_MODEL:
        raise ValueError(
            "OPENROUTER_MODEL is not configured."
        )

    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    system_prompt = """
You are the natural-language interface for a diabetes
prediction machine learning application.

Your job is to understand the user's request and decide
whether enough valid, unambiguous patient information exists
to make a prediction.

Return ONLY valid JSON.

Use exactly this structure:

{
    "status": "valid | needs_clarification | out_of_scope",
    "patient": {
        "gender": null,
        "age": null,
        "hypertension": null,
        "heart_disease": null,
        "smoking_history": null,
        "bmi": null,
        "HbA1c_level": null,
        "blood_glucose_level": null
    },
    "issues": [
        {
            "field": "field_name",
            "message": "clear explanation of the problem"
        }
    ],
    "message": "conversational response for the user"
}

The required patient fields are:

gender
age
hypertension
heart_disease
smoking_history
bmi
HbA1c_level
blood_glucose_level


VALID VALUES

gender must be exactly one of:

Female
Male
Other

hypertension must be:

0 = does not have hypertension
1 = has hypertension

heart_disease must be:

0 = does not have heart disease
1 = has heart disease

smoking_history must be exactly one of:

never
No Info
current
former
ever
not current

age must be numeric.

bmi must be numeric.

HbA1c_level must be numeric.

blood_glucose_level must be numeric.


STATUS RULES

Use "valid" only when all required patient fields are
present, valid, and unambiguous.

Use "needs_clarification" when:

- one or more required fields are missing
- a value is unclear
- a value is contradictory
- a value is invalid
- multiple possible interpretations exist

When there are multiple problems, include EVERY problem
in the issues list.

Do not stop after finding the first problem.

For every problem, identify the exact field involved.

If information contradicts itself, explicitly say that
the field is contradictory and explain what conflicted.

Do not guess which conflicting statement is correct.

Use "out_of_scope" when the user is asking for something
that is not a patient prediction request, such as:

- treatment recommendations
- medication advice
- diagnosis
- general medical advice
- unrelated questions

For out-of-scope requests, respond politely and
conversationally.

Explain that this application can help process patient
information for the diabetes prediction model.

Do not sound like an error message.

Never provide medical advice.


IMPORTANT RULES

Never invent missing patient information.

Never resolve contradictions by guessing.

If a required feature is missing or uncertain,
set that feature to null.

If several features have problems, identify all of them.

Do not make the diabetes prediction yourself.

Do not include markdown.

Return only JSON.
"""

    response = (
        client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            temperature=0,
        )
    )

    content = (
        response.choices[0]
        .message.content
    )

    if not content:
        raise ValueError(
            "The LLM returned an empty response."
        )

    try:
        structured_data = json.loads(
            content
        )

    except json.JSONDecodeError as exc:
        raise ValueError(
            "The LLM returned invalid JSON."
        ) from exc

    # ========================================================
    # Read LLM response structure
    # ========================================================

    status = structured_data.get(
        "status"
    )

    message = structured_data.get(
        "message",
        "",
    )

    issues = structured_data.get(
        "issues",
        [],
    )

    patient_data = structured_data.get(
        "patient"
    )

    # ========================================================
    # Out-of-scope request
    # ========================================================

    if status == "out_of_scope":

        return {
            "status": "out_of_scope",
            "message": message,
            "issues": [],
            "patient": None,
        }

    # ========================================================
    # Missing, contradictory, ambiguous, or invalid input
    # ========================================================

    if status == "needs_clarification":

        return {
            "status": "needs_clarification",
            "message": message,
            "issues": issues,
            "patient": patient_data,
        }

    # ========================================================
    # Valid patient
    # ========================================================

    if status == "valid":

        if patient_data is None:

            raise ValueError(
                "The LLM marked the request as valid "
                "but did not return patient information."
            )

        patient = parse_structured_input(
            patient_data
        )

        return {
            "status": "valid",
            "message": message,
            "issues": [],
            "patient": patient,
        }

    # ========================================================
    # Unknown response type
    # ========================================================

    raise ValueError(
        "The LLM returned an unknown request status."
    )


def validate_patient_features(patient):
    """
    Validate parsed diabetes-model feature values.
    """

    if patient["gender"] not in VALID_GENDERS:
        raise ValueError(
            "Invalid gender value."
        )

    if patient["smoking_history"] not in (
        VALID_SMOKING_HISTORY
    ):
        raise ValueError(
            "Invalid smoking_history value."
        )

    if patient["age"] < 0:
        raise ValueError(
            "Age cannot be negative."
        )

    if patient["bmi"] <= 0:
        raise ValueError(
            "BMI must be greater than zero."
        )

    if patient["HbA1c_level"] <= 0:
        raise ValueError(
            "HbA1c_level must be greater than zero."
        )

    if patient["blood_glucose_level"] <= 0:
        raise ValueError(
            "Blood glucose level must be greater than zero."
        )

    if patient["hypertension"] not in {0, 1}:
        raise ValueError(
            "Hypertension must be 0 or 1."
        )

    if patient["heart_disease"] not in {0, 1}:
        raise ValueError(
            "Heart disease must be 0 or 1."
        )

    return True


def process_patient_request(text):

    parsed = parse_natural_language_input(
        text
    )

    status = parsed["status"]

    if status == "out_of_scope":

        return {
            "status": "out_of_scope",
            "patient": None,
            "result": None,
            "issues": [],
            "response": parsed["message"],
        }

    if status == "needs_clarification":

        return {
            "status": "needs_clarification",
            "patient": parsed["patient"],
            "result": None,
            "issues": parsed["issues"],
            "response": parsed["message"],
        }

    patient = parsed["patient"]

    result = predict_diabetes(
        patient
    )

    response = format_prediction_response(
        result
    )

    return {
        "status": "valid",
        "patient": patient,
        "result": result,
        "issues": [],
        "response": response,
    }


def patient_to_dataframe(patient):
    """
    Convert a validated patient dictionary into a one-row dataframe.
    """

    return pd.DataFrame(
        [patient],
        columns=REQUIRED_FEATURES,
    )


def load_model(model_path=MODEL_PATH):
    """
    Load the trained diabetes prediction pipeline.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}"
        )

    with open(model_path, "rb") as file:
        model = cloudpickle.load(file)

    return model


def load_best_threshold(
    selection_path=SELECTION_PATH,
):
    """
    Load the classification threshold selected
    during model validation.
    """

    if not selection_path.exists():
        raise FileNotFoundError(
            "Model selection metadata not found at "
            f"{selection_path}"
        )

    with open(
        selection_path,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    return float(
        metadata["best_threshold"]
    )


def predict_diabetes(
    patient,
    model=None,
    threshold=None,
):
    """
    Generate a diabetes prediction and probability.
    """

    validate_patient_features(patient)

    patient_df = patient_to_dataframe(
        patient
    )

    if model is None:
        model = load_model()

    if threshold is None:
        threshold = load_best_threshold()

    probability = (
        model.predict_proba(
            patient_df
        )[:, 1][0]
    )

    prediction = int(
        probability >= threshold
    )

    return {
        "prediction": prediction,
        "probability": float(probability),
        "threshold": float(threshold),
    }


def get_prediction_context(
    probability,
    threshold,
):
    difference = probability - threshold

    if difference <= -0.10:
        return (
            "The prediction is well below "
            "the classification threshold."
        )

    elif difference <= -0.05:
        return (
            "The prediction is below "
            "the classification threshold."
        )

    elif difference < 0.05:
        return (
            "The prediction is very close "
            "to the classification threshold."
        )

    elif difference < 0.10:
        return (
            "The prediction is above "
            "the classification threshold."
        )

    else:
        return (
            "The prediction is well above "
            "the classification threshold."
        )


def format_prediction_response(result):
    """
    Create a user-facing response that explains
    the model prediction relative to its threshold.
    """

    probability = result["probability"]
    threshold = result["threshold"]

    probability_percent = (
        probability * 100
    )

    threshold_percent = (
        threshold * 100
    )

    difference = (
        probability - threshold
    )

    difference_percent = (
        abs(difference) * 100
    )

    if result["prediction"] == 1:
        prediction_text = (
            "The model classifies this case "
            "as above the diabetes prediction threshold."
        )
    else:
        prediction_text = (
            "The model classifies this case "
            "as below the diabetes prediction threshold."
        )

    if difference <= -0.10:
        context_text = (
            "The estimated probability is well below "
            "the model's classification threshold."
        )

    elif difference <= -0.05:
        context_text = (
            "The estimated probability is below "
            "the model's classification threshold."
        )

    elif difference < 0.05:
        context_text = (
            "The estimated probability is very close "
            "to the model's classification threshold."
        )

    elif difference < 0.10:
        context_text = (
            "The estimated probability is above "
            "the model's classification threshold."
        )

    else:
        context_text = (
            "The estimated probability is well above "
            "the model's classification threshold."
        )

    return (
        f"{prediction_text} "
        f"Estimated model probability: "
        f"{probability_percent:.1f}%. "
        f"Classification threshold: "
        f"{threshold_percent:.1f}%. "
        f"{context_text} "
        f"The prediction is "
        f"{difference_percent:.1f} percentage points "
        f"from the decision threshold. "
        "This prediction is for educational purposes "
        "and is not a medical diagnosis. "
        "Please consult a healthcare professional "
        "for personalized medical advice."
    )


if __name__ == "__main__":

    user_text = """
    I am a 58-year-old male.
    I have hypertension but no heart disease.
    I used to smoke.
    My BMI is 30.5.
    My HbA1c is 6.7.
    My blood glucose level is 160.
    """

    output = process_patient_request(
        user_text
    )

    print(
        json.dumps(
            output,
            indent=2,
        )
    )

