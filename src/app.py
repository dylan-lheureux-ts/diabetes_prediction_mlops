import json
import cloudpickle
from pathlib import Path
import os

import pandas as pd
import xgboost as xgb
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
    Validate structured patient information and report every
    problem found in one response.
    """

    if not isinstance(data, dict):
        raise ValueError(
            "Parsed patient information must be a dictionary."
        )

    issues = []
    patient = {}

    # Categorical fields
    for field in ["gender", "smoking_history"]:
        value = data.get(field)

        if value is None:
            issues.append(
                f"{field}: missing required value"
            )
        else:
            patient[field] = str(value)

    # Numeric fields
    numeric_fields = [
        "age",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level",
    ]

    for field in numeric_fields:
        value = data.get(field)

        if value is None:
            issues.append(
                f"{field}: missing required value"
            )
            continue

        try:
            patient[field] = float(value)
        except (TypeError, ValueError):
            issues.append(
                f"{field}: must be a numeric value"
            )

    # Binary fields
    for field in ["hypertension", "heart_disease"]:
        value = data.get(field)

        if value is None:
            issues.append(
                f"{field}: missing required value"
            )
            continue

        try:
            numeric_value = float(value)

            if numeric_value not in {0.0, 1.0}:
                raise ValueError

            patient[field] = int(numeric_value)

        except (TypeError, ValueError):
            issues.append(
                f"{field}: must be 0 or 1"
            )

    # Validate categorical values
    if (
        "gender" in patient
        and patient["gender"] not in VALID_GENDERS
    ):
        issues.append(
            "gender: must be Female, Male, or Other"
        )

    if (
        "smoking_history" in patient
        and patient["smoking_history"]
        not in VALID_SMOKING_HISTORY
    ):
        issues.append(
            "smoking_history: invalid smoking history"
        )

    # Validate numeric ranges
    if "age" in patient and patient["age"] < 0:
        issues.append(
            "age: cannot be negative"
        )

    if "bmi" in patient and patient["bmi"] <= 0:
        issues.append(
            "bmi: must be greater than zero"
        )

    if (
        "HbA1c_level" in patient
        and patient["HbA1c_level"] <= 0
    ):
        issues.append(
            "HbA1c_level: must be greater than zero"
        )

    if (
        "blood_glucose_level" in patient
        and patient["blood_glucose_level"] <= 0
    ):
        issues.append(
            "blood_glucose_level: must be greater than zero"
        )

    if issues:
        raise ValueError(
            "Please correct the following patient information: "
            + "; ".join(issues)
        )

    return patient

def parse_natural_language_input(text):
    """
    Use the LLM to classify the request and, when appropriate,
    extract the patient features required by the model.
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
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )

    system_prompt = """
You are the natural-language interface for a diabetes
prediction machine learning application.

Return ONLY valid JSON using exactly this structure:

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

Required features:
gender, age, hypertension, heart_disease, smoking_history,
bmi, HbA1c_level, blood_glucose_level.

Valid gender values:
Female, Male, Other.

hypertension and heart_disease:
0 = no, 1 = yes.

Valid smoking_history values:
never, No Info, current, former, ever, not current.

Use "valid" only when every required feature is present,
valid, and unambiguous.

Use "needs_clarification" when any feature is missing,
invalid, ambiguous, or contradictory.

When multiple features have problems, include EVERY problem
in the issues list. Never stop after the first problem.

For contradictions, identify the exact field, explicitly say
the information is contradictory, and explain what conflicted.
Set the contradictory field to null. Never guess which
statement is correct.

Use "out_of_scope" for requests this model cannot answer,
including treatment recommendations, medication advice,
diagnosis, or unrelated questions.

For out-of-scope requests, respond politely and
conversationally. Explain that this application can make and
explain a diabetes model prediction from the eight required
features, but cannot provide a medical diagnosis or treatment.

Never invent missing patient information.
Never provide medical advice.
Do not make the diabetes prediction yourself.
Do not include markdown.
Return only JSON.
"""

    response = client.chat.completions.create(
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

    content = response.choices[0].message.content

    if not content:
        raise ValueError(
            "The LLM returned an empty response."
        )

    try:
        structured_data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "The LLM returned invalid JSON."
        ) from exc

    if not isinstance(structured_data, dict):
        raise ValueError(
            "The LLM returned an invalid response structure."
        )

    status = structured_data.get("status")
    message = structured_data.get("message", "")
    issues = structured_data.get("issues", [])
    patient_data = structured_data.get("patient")

    if status == "out_of_scope":
        return {
            "status": "out_of_scope",
            "message": message,
            "issues": [],
            "patient": None,
        }

    if status == "needs_clarification":
        return {
            "status": "needs_clarification",
            "message": message,
            "issues": issues,
            "patient": patient_data,
        }

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
    """
    Process a natural-language request from extraction through
    prediction, local model explanation, and LLM response.
    """

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
            "feature_contributions": [],
            "response": parsed["message"],
        }

    if status == "needs_clarification":
        return {
            "status": "needs_clarification",
            "patient": parsed["patient"],
            "result": None,
            "issues": parsed["issues"],
            "feature_contributions": [],
            "response": parsed["message"],
        }

    patient = parsed["patient"]
    model = load_model()

    result = predict_diabetes(
        patient,
        model=model,
    )

    feature_contributions = explain_prediction(
        patient,
        model=model,
    )

    response = generate_prediction_response(
        patient,
        result,
        feature_contributions,
    )

    return {
        "status": "valid",
        "patient": patient,
        "result": result,
        "issues": [],
        "feature_contributions": feature_contributions,
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


def _original_feature_name(transformed_name):
    """
    Map a transformed ColumnTransformer feature name back to
    one of the eight original patient features.
    """

    clean_name = transformed_name.split(
        "__",
        1,
    )[-1]

    for feature in sorted(
        REQUIRED_FEATURES,
        key=len,
        reverse=True,
    ):
        if (
            clean_name == feature
            or clean_name.startswith(
                feature + "_"
            )
        ):
            return feature

    return clean_name


def explain_prediction(
    patient,
    model=None,
    top_n=8,
):
    """
    Calculate local XGBoost feature contributions for one
    patient and aggregate one-hot encoded columns back to the
    original patient features.

    Contribution values are model-attribution values in the
    XGBoost margin space. They describe model influence, not
    medical causation.
    """

    validate_patient_features(patient)

    if model is None:
        model = load_model()

    if not hasattr(model, "named_steps"):
        return []

    preprocessor = model.named_steps.get(
        "preprocessor"
    )

    classifier = model.named_steps.get(
        "classifier"
    )

    if (
        preprocessor is None
        or classifier is None
        or not hasattr(
            classifier,
            "get_booster",
        )
    ):
        return []

    patient_df = patient_to_dataframe(
        patient
    )

    transformed = preprocessor.transform(
        patient_df
    )

    feature_names = list(
        preprocessor.get_feature_names_out()
    )

    booster = classifier.get_booster()

    contributions = booster.predict(
        xgb.DMatrix(transformed),
        pred_contribs=True,
    )[0]

    # XGBoost appends the bias/base value as the last item.
    feature_values = contributions[:-1]

    if len(feature_values) != len(feature_names):
        raise ValueError(
            "Feature contribution output did not match "
            "the transformed feature names."
        )

    aggregated = {}

    for transformed_name, contribution in zip(
        feature_names,
        feature_values,
    ):
        original_feature = _original_feature_name(
            transformed_name
        )

        aggregated[original_feature] = (
            aggregated.get(
                original_feature,
                0.0,
            )
            + float(contribution)
        )

    total_absolute = sum(
        abs(value)
        for value in aggregated.values()
    )

    ranked = sorted(
        aggregated.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    explanation = []

    for feature, contribution in ranked[:top_n]:
        if contribution > 0:
            direction = "increased"
        elif contribution < 0:
            direction = "decreased"
        else:
            direction = "neutral"

        if total_absolute:
            share = (
                abs(contribution)
                / total_absolute
                * 100
            )
        else:
            share = 0.0

        explanation.append(
            {
                "feature": feature,
                "value": patient.get(feature),
                "direction": direction,
                "contribution": float(
                    contribution
                ),
                "relative_influence_percent": float(
                    share
                ),
            }
        )

    return explanation


def generate_prediction_response(
    patient,
    result,
    feature_contributions,
):
    """
    Ask the LLM to explain the model's completed prediction.
    The model result and feature contributions are supplied as
    fixed facts that the LLM must not change.
    """

    if not OPENROUTER_API_KEY or not OPENROUTER_MODEL:
        return format_prediction_response(
            result
        )

    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )

    grounded_data = {
        "patient": patient,
        "prediction": result["prediction"],
        "probability": result["probability"],
        "threshold": result["threshold"],
        "feature_contributions": (
            feature_contributions
        ),
    }

    system_prompt = """
You explain the output of a diabetes prediction machine
learning model.

The supplied model result is a fixed fact. Never change,
recalculate, or contradict the prediction, probability, or
classification threshold.

Explain the result clearly and conversationally.

When feature contributions are provided, you may explain
which patient features had the greatest influence on THIS
model prediction and whether each pushed the model output up
or down.

Feature contributions describe how the model used the input.
They do NOT establish medical causation, diagnosis, or true
clinical risk.

If the user or data refers to a "diagnosis", gently clarify
that this is a model prediction, not a diagnosis.

Include relevant caveats and say that the result is for
educational purposes and is not medical advice or a medical
diagnosis.

Do not recommend medications or treatments.
Do not invent additional medical facts.
Use plain text, not markdown.
"""

    try:
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        grounded_data
                    ),
                },
            ],
            temperature=0.2,
        )

        content = (
            response.choices[0]
            .message.content
        )

        if content:
            return content.strip()

    except Exception:
        pass

    return format_prediction_response(
        result
    )


def answer_prediction_question(
    question,
    patient,
    result,
    feature_contributions,
):
    """
    Answer a follow-up question using only the completed model
    prediction and its local feature contributions.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError(
            "Please enter a question about the prediction."
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
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )

    context = {
        "patient": patient,
        "prediction": result,
        "feature_contributions": (
            feature_contributions
        ),
    }

    system_prompt = """
You answer follow-up questions about one completed diabetes
machine learning model prediction.

Use ONLY the supplied prediction data and feature
contributions. Do not invent model behavior or medical facts.

You may answer questions such as:
- Which feature influenced my prediction the most?
- What were the top three factors?
- Did BMI push the prediction up or down?
- Why was the prediction above or below the threshold?

If the user says "diagnosis", gently clarify that the
application produced a model prediction, not a diagnosis.

Feature contributions describe model influence only. They do
not prove that a feature caused diabetes.

If the user asks for diagnosis, treatment, medication, or
other medical advice, politely explain that you can explain
the model prediction but cannot provide medical advice.

Keep the answer conversational and concise.
Use plain text, not markdown.
"""

    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    "Prediction context:\n"
                    + json.dumps(context)
                    + "\n\nUser question:\n"
                    + question
                ),
            },
        ],
        temperature=0.2,
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError(
            "The LLM returned an empty response."
        )

    return content.strip()


def get_prediction_context(

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

