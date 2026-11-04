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
    Convert structured input into a validated patient feature dictionary.

    This function is intentionally separate from the live LLM call so it
    can be tested without requiring an API connection.
    """

    if not isinstance(data, dict):
        raise ValueError(
            "Parsed input must be a dictionary."
        )

    missing = [
        feature
        for feature in REQUIRED_FEATURES
        if feature not in data
        or data[feature] is None
    ]

    if missing:
        raise ValueError(
            "Missing required features: "
            + ", ".join(missing)
        )

    patient = {
        "gender": str(data["gender"]),
        "age": float(data["age"]),
        "hypertension": int(data["hypertension"]),
        "heart_disease": int(data["heart_disease"]),
        "smoking_history": str(
            data["smoking_history"]
        ),
        "bmi": float(data["bmi"]),
        "HbA1c_level": float(
            data["HbA1c_level"]
        ),
        "blood_glucose_level": float(
            data["blood_glucose_level"]
        ),
    }

    validate_patient_features(patient)

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
You are a structured data extraction assistant.

Your only task is to extract patient information
for a diabetes prediction model.

Return ONLY a valid JSON object.

The required fields are:

gender
age
hypertension
heart_disease
smoking_history
bmi
HbA1c_level
blood_glucose_level

Rules:

gender must be exactly one of:
Female
Male
Other

hypertension must be:
0 if the patient does not have hypertension
1 if the patient has hypertension

heart_disease must be:
0 if the patient does not have heart disease
1 if the patient has heart disease

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

Never guess or invent medical information.

If a required feature is not provided or cannot
be determined confidently, return null for it.

Do not make a diabetes prediction.

Do not provide medical advice.

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

    return parse_structured_input(
        structured_data
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
    """
    Process a natural-language patient request from
    extraction through prediction.
    """

    patient = parse_natural_language_input(
        text
    )

    result = predict_diabetes(
        patient
    )

    response = format_prediction_response(
        result
    )

    return {
        "patient": patient,
        "result": result,
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


def format_prediction_response(result):
    """
    Create a basic user-facing response.

    The LLM layer can later expand this into a more
    conversational explanation.
    """

    probability_percent = (
        result["probability"] * 100
    )

    if result["prediction"] == 1:
        prediction_text = (
            "The model predicts a higher likelihood "
            "of diabetes."
        )
    else:
        prediction_text = (
            "The model predicts a lower likelihood "
            "of diabetes."
        )

    return (
        f"{prediction_text} "
        f"Estimated model probability: "
        f"{probability_percent:.1f}%. "
        "This prediction is for educational purposes "
        "and is not a medical diagnosis. Please consult a "
        "healthcare professional for personalized medical "
        "advice."
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

