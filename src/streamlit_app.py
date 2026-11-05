import streamlit as st

from src.app import process_patient_request


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


# ============================================================
# Helper functions
# ============================================================

def use_example():
    """
    Load the example patient into the text box.
    """

    st.session_state.patient_text = EXAMPLE_TEXT
    st.session_state.prediction_output = None


def clear_input():
    """
    Clear the patient description and previous result.
    """

    st.session_state.patient_text = ""
    st.session_state.prediction_output = None


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


# ============================================================
# Input buttons
# ============================================================

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

            st.write(
                "Please make sure the description "
                "contains all required patient information."
            )

        except FileNotFoundError as exc:

            st.session_state.prediction_output = None

            st.error(
                "A required project file could not "
                "be found."
            )

            st.code(
                str(exc)
            )

            st.write(
                "Make sure the dataset and model "
                "artifacts have been restored with DVC."
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
# Display prediction
# ============================================================

output = st.session_state.prediction_output

if output is not None:

    patient = output["patient"]
    result = output["result"]
    response = output["response"]

    probability = result["probability"]
    threshold = result["threshold"]
    prediction = result["prediction"]

    probability_percent = (
        probability * 100
    )

    threshold_percent = (
        threshold * 100
    )

    st.divider()

    st.success(
        "Prediction completed successfully."
    )

    st.subheader(
        "Prediction Result"
    )


    # ========================================================
    # Main metrics
    # ========================================================

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


    # ========================================================
    # Classification
    # ========================================================

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


    # ========================================================
    # Explanation
    # ========================================================

    st.subheader(
        "Model Explanation"
    )

    st.write(
        response
    )


    # ========================================================
    # Extracted patient information
    # ========================================================

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
        "If a required value is missing, the "
        "application will ask for more complete "
        "information rather than guessing."
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
        2. The LLM extracts the eight required features.
        3. The extracted values are validated.
        4. The trained XGBoost model generates a probability.
        5. The probability is compared with the selected
           classification threshold.
        6. The application explains the result relative
           to that threshold.
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