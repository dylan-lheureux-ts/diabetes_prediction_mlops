import streamlit as st

from src.app import (
    answer_prediction_question,
    process_patient_request,
)


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Diabetes Prediction Assistant",
    page_icon="🩺",
    layout="centered",
)


# ============================================================
# Example patient
# ============================================================

EXAMPLE_TEXT = """
I am a 58-year-old male.
I have hypertension but no heart disease.
I used to smoke.
My BMI is 30.5.
My HbA1c is 6.7.
My blood glucose level is 160.
""".strip()


# ============================================================
# Session state
# ============================================================

if "patient_text" not in st.session_state:
    st.session_state.patient_text = ""

if "prediction_output" not in st.session_state:
    st.session_state.prediction_output = None

if "followup_question" not in st.session_state:
    st.session_state.followup_question = ""

if "followup_answer" not in st.session_state:
    st.session_state.followup_answer = None


# ============================================================
# Helper functions
# ============================================================

def use_example():
    st.session_state.patient_text = EXAMPLE_TEXT
    st.session_state.prediction_output = None
    st.session_state.followup_question = ""
    st.session_state.followup_answer = None


def clear_input():
    st.session_state.patient_text = ""
    st.session_state.prediction_output = None
    st.session_state.followup_question = ""
    st.session_state.followup_answer = None


# ============================================================
# Page header
# ============================================================

st.title("Diabetes Prediction Assistant")

st.write(
    """
    Describe a patient in plain English and the application
    will extract the required information, run the diabetes
    prediction model, and explain the result.
    """
)

st.info(
    "This application is for educational purposes only "
    "and is not a medical diagnostic tool."
)


# ============================================================
# User input
# ============================================================

st.text_area(
    "Describe the patient",
    key="patient_text",
    placeholder=(
        "Example: I am a 58-year-old male. "
        "I have hypertension but no heart disease..."
    ),
    height=180,
)

example_col, clear_col = st.columns(2)

with example_col:
    st.button(
        "Use Example",
        on_click=use_example,
        use_container_width=True,
    )

with clear_col:
    st.button(
        "Clear",
        on_click=clear_input,
        use_container_width=True,
    )


# ============================================================
# Generate prediction
# ============================================================

if st.button(
    "Generate Prediction",
    type="primary",
    use_container_width=True,
):
    patient_text = (
        st.session_state.patient_text.strip()
    )

    st.session_state.followup_question = ""
    st.session_state.followup_answer = None

    if not patient_text:
        st.warning(
            "Please enter a patient description first."
        )
        st.session_state.prediction_output = None

    else:
        try:
            with st.spinner(
                "Processing patient information..."
            ):
                output = process_patient_request(
                    patient_text
                )

            st.session_state.prediction_output = (
                output
            )

        except ValueError as exc:
            st.session_state.prediction_output = None

            st.error(
                f"Unable to process the request: {exc}"
            )

        except FileNotFoundError as exc:
            st.session_state.prediction_output = None

            st.error(
                "A required project file could not be found."
            )

            st.code(
                str(exc)
            )

            st.write(
                "Make sure the dataset and model artifacts "
                "have been restored with DVC."
            )

        except Exception as exc:
            st.session_state.prediction_output = None

            st.error(
                "Something unexpected happened while "
                "processing the request."
            )

            with st.expander(
                "View error details"
            ):
                st.exception(exc)


# ============================================================
# Display request result
# ============================================================

output = st.session_state.prediction_output

if output is not None:
    status = output.get(
        "status",
        "valid",
    )

    st.divider()

    # ========================================================
    # Out-of-scope request
    # ========================================================

    if status == "out_of_scope":
        st.info(
            output["response"]
        )

    # ========================================================
    # Clarification required
    # ========================================================

    elif status == "needs_clarification":
        st.warning(
            output["response"]
            or (
                "I need a little more information "
                "before I can make a prediction."
            )
        )

        issues = output.get(
            "issues",
            [],
        )

        if issues:
            st.subheader(
                "Information to Fix"
            )

            for issue in issues:
                field = issue.get(
                    "field",
                    "patient information",
                )

                message = issue.get(
                    "message",
                    "Please review this value.",
                )

                st.write(
                    f"**{field}:** {message}"
                )

    # ========================================================
    # Valid prediction
    # ========================================================

    elif status == "valid":
        patient = output["patient"]
        result = output["result"]
        response = output["response"]
        feature_contributions = output.get(
            "feature_contributions",
            [],
        )

        probability = result["probability"]
        threshold = result["threshold"]
        prediction = result["prediction"]

        probability_percent = (
            probability * 100
        )

        threshold_percent = (
            threshold * 100
        )

        st.success(
            "Prediction completed successfully."
        )

        st.subheader(
            "Prediction Result"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Model Probability",
                f"{probability_percent:.1f}%",
            )

        with col2:
            st.metric(
                "Classification Threshold",
                f"{threshold_percent:.1f}%",
            )

        if prediction == 1:
            st.warning(
                "The model classifies this case as "
                "above the diabetes prediction threshold."
            )
        else:
            st.success(
                "The model classifies this case as "
                "below the diabetes prediction threshold."
            )

        # ====================================================
        # LLM-generated explanation
        # ====================================================

        st.subheader(
            "Model Explanation"
        )

        st.write(
            response
        )

        # ====================================================
        # Local model contributions
        # ====================================================

        if feature_contributions:
            with st.expander(
                "What influenced this prediction?"
            ):
                st.write(
                    "These values describe how strongly each "
                    "feature influenced this specific model "
                    "prediction. They do not show medical "
                    "causation or clinical importance."
                )

                for item in feature_contributions:
                    label = (
                        item["feature"]
                        .replace("_", " ")
                        .title()
                    )

                    direction = item[
                        "direction"
                    ]

                    influence = item[
                        "relative_influence_percent"
                    ]

                    st.write(
                        f"**{label}:** {direction} the model "
                        f"output — {influence:.1f}% of the "
                        "total local influence."
                    )

        # ====================================================
        # Extracted patient information
        # ====================================================

        with st.expander(
            "View Extracted Patient Information"
        ):
            patient_data = {
                "Gender": patient["gender"],
                "Age": patient["age"],
                "Hypertension": (
                    "Yes"
                    if patient["hypertension"] == 1
                    else "No"
                ),
                "Heart Disease": (
                    "Yes"
                    if patient["heart_disease"] == 1
                    else "No"
                ),
                "Smoking History": (
                    patient["smoking_history"]
                ),
                "BMI": patient["bmi"],
                "HbA1c Level": (
                    patient["HbA1c_level"]
                ),
                "Blood Glucose Level": (
                    patient["blood_glucose_level"]
                ),
            }

            for label, value in patient_data.items():
                st.write(
                    f"**{label}:** {value}"
                )

        # ====================================================
        # Follow-up question about this prediction
        # ====================================================

        st.subheader(
            "Ask About This Prediction"
        )

        st.text_input(
            "Ask a follow-up question",
            key="followup_question",
            placeholder=(
                "Example: Which feature played the biggest "
                "role in this prediction?"
            ),
        )

        if st.button(
            "Ask About Prediction",
            use_container_width=True,
        ):
            question = (
                st.session_state
                .followup_question
                .strip()
            )

            if not question:
                st.warning(
                    "Please enter a follow-up question."
                )

            else:
                try:
                    with st.spinner(
                        "Explaining the prediction..."
                    ):
                        answer = (
                            answer_prediction_question(
                                question,
                                patient,
                                result,
                                feature_contributions,
                            )
                        )

                    st.session_state.followup_answer = (
                        answer
                    )

                except Exception as exc:
                    st.error(
                        "I could not answer that follow-up "
                        "question."
                    )

                    with st.expander(
                        "View error details"
                    ):
                        st.exception(exc)

        if st.session_state.followup_answer:
            st.info(
                st.session_state.followup_answer
            )


# ============================================================
# Required patient information
# ============================================================

with st.expander(
    "What information does the model need?"
):
    st.write(
        """
        The patient description should include all
        eight of the following features:
        """
    )

    st.markdown(
        """
        - **Gender**
        - **Age**
        - **Hypertension status**
        - **Heart disease status**
        - **Smoking history**
        - **BMI**
        - **HbA1c level**
        - **Blood glucose level**
        """
    )

    st.write(
        "If a required value is missing, contradictory, "
        "or invalid, the application will identify the "
        "specific information that needs to be corrected."
    )


# ============================================================
# How the application works
# ============================================================

with st.expander(
    "How does this application work?"
):
    st.markdown(
        """
        1. You describe a patient in plain English.
        2. The LLM extracts and validates the eight required features.
        3. The trained XGBoost model generates a probability.
        4. The probability is compared with the selected threshold.
        5. XGBoost local feature contributions identify which inputs
           influenced this specific prediction.
        6. The LLM explains the fixed model result and can answer
           grounded follow-up questions about the prediction.
        """
    )


# ============================================================
# Disclaimer
# ============================================================

st.divider()

st.caption(
    "Disclaimer: This project was created for educational "
    "and portfolio purposes. It is not a medical device and "
    "should not be used to diagnose diabetes or make "
    "healthcare decisions. Consult a qualified healthcare "
    "professional for medical guidance."
)
