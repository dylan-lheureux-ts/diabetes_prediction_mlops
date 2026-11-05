# Diabetes Prediction MLOps Project

## Project Overview

This project is an end-to-end machine learning application that predicts whether a patient is likely to have diabetes based on several health and lifestyle features.

The main goal of this project was not just to train a model, but to build a more complete machine learning workflow. The project includes data preprocessing, model experimentation, MLflow tracking, model selection, DVC data and model versioning, automated testing, GitHub Actions, and a natural-language interface powered by an LLM.

A user can describe a patient in plain English, and the application uses an LLM to convert that description into the structured features required by the machine learning model. The selected model then returns a probability and classification based on the threshold chosen during model development.

This project is for educational purposes only and is not intended to provide medical diagnoses or replace professional medical advice.

---

## Problem Statement

Diabetes prediction is a binary classification problem.

The goal of the machine learning model is to predict:

- `0` - No diabetes
- `1` - Diabetes

The dataset is imbalanced, with diabetes cases representing a much smaller percentage of the data. Because of this, I did not want to rely only on accuracy when comparing models.

I also evaluated:

- Precision
- Recall
- F1 Score
- F-beta Score
- AUC-ROC
- Average Precision

Average Precision was especially useful because it gives a better picture of model performance when the positive class is less common.

---

## Dataset

The dataset contains patient health and lifestyle information used to predict diabetes.

The original dataset contains approximately 100,000 rows.

During preprocessing, duplicate records were removed, leaving approximately 96,000 unique records.

The model uses the following features:

| Feature               | Description                           |
| --------------------- | ------------------------------------- |
| `gender`              | Patient gender                        |
| `age`                 | Patient age                           |
| `hypertension`        | Whether the patient has hypertension  |
| `heart_disease`       | Whether the patient has heart disease |
| `smoking_history`     | Patient smoking history               |
| `bmi`                 | Body Mass Index                       |
| `HbA1c_level`         | HbA1c blood test level                |
| `blood_glucose_level` | Blood glucose level                   |

The target variable is:

```text
diabetes
```

The dataset is versioned using DVC instead of being stored directly in Git.

---

## Data Preprocessing

The preprocessing pipeline was built using Scikit-Learn pipelines and a `ColumnTransformer`.

Numeric features are:

```text
age
bmi
HbA1c_level
blood_glucose_level
```

Numeric values are:

- Filled using the median when missing
- Standardized using `StandardScaler`

Categorical features are:

```text
gender
smoking_history
```

Categorical values are:

- Filled using the most frequent value when missing
- Encoded using `OneHotEncoder`

Binary features are:

```text
hypertension
heart_disease
```

These are validated to make sure their values are either `0` or `1`.

Using a Scikit-Learn pipeline keeps preprocessing and model training together and helps prevent data leakage between the training, validation, and test sets.

---

## Train, Validation, and Test Strategy

I separated the data into training, validation, and held-out test sets.

The approximate split is:

```text
Training:   60%
Validation: 20%
Test:       20%
```

The training set is used to fit the models.

The validation set is used to compare models and select the classification threshold.

The final test set is kept separate during model development and is only used for the final evaluation.

This was important because repeatedly evaluating models on the test set could cause decisions to become indirectly tuned to the test data.

---

## Models Tested

I tested multiple configurations across three different classification algorithms:

- Logistic Regression
- Random Forest
- XGBoost

The experiment sweep includes multiple hyperparameter configurations for each model family.

For example, the XGBoost experiments tested different combinations of:

- Number of estimators
- Learning rate
- Maximum tree depth
- Subsampling
- Feature subsampling

All experiments are tracked using MLflow.

---

## MLflow Experiment Tracking

MLflow is used to keep track of model experiments.

For each run, the project logs information such as:

- Model type
- Hyperparameters
- Training configuration
- Dataset information
- Classification threshold
- Accuracy
- Precision
- Recall
- F1 Score
- F-beta Score
- AUC-ROC
- Average Precision
- Trained model artifact

The project also uses:

```python
mlflow.search_runs()
```

to compare completed experiments programmatically.

Instead of manually deciding which model was best, the sweep script searches the completed MLflow runs and selects the model with the highest validation Average Precision.

---

## Model Selection

The final selected model was an XGBoost classifier.

The model itself and the classification threshold are treated as two separate decisions.

Average Precision is used to help select the best model configuration.

The classification threshold is selected on the validation data using an F-beta score.

This allows the model probability ranking and the final classification decision to be evaluated separately.

The selected classification threshold was:

```text
0.25
```

This means a predicted probability of 25% or higher is classified as being above the model's diabetes prediction threshold.

The threshold should not be interpreted as a medical risk threshold. It is simply the decision boundary selected for this machine learning model.

---

## Final Test Results

After model development was complete, the selected XGBoost model was evaluated on the held-out test set.

The test set contained approximately 19,230 records.

| Metric            | Final Test Result |
| ----------------- | ----------------: |
| Accuracy          |            0.9613 |
| Precision         |            0.7835 |
| Recall            |            0.7748 |
| F1 Score          |            0.7791 |
| F-beta Score      |            0.7772 |
| AUC-ROC           |            0.9779 |
| Average Precision |            0.8850 |

These results show that the model performs well overall while still maintaining a reasonable balance between precision and recall on the minority diabetes class.

Accuracy alone would not tell the full story because most patients in the dataset do not have diabetes. For that reason, I focused more heavily on metrics such as recall, precision, F1, F-beta, AUC-ROC, and Average Precision.

---

## Natural-Language LLM Interface

One of the main features of this project is the natural-language interface.

Instead of requiring the user to manually create a structured dictionary, a user can enter something like:

```text
I am a 58-year-old male.
I have hypertension but no heart disease.
I used to smoke.
My BMI is 30.5.
My HbA1c is 6.7.
My blood glucose level is 160.
```

The application sends the description to an LLM through OpenRouter.

The LLM is instructed to extract only the eight features required by the machine learning model.

The workflow is:

```text
Natural-language patient description
                ↓
        OpenRouter LLM
                ↓
Structured patient information
                ↓
        Input validation
                ↓
     XGBoost prediction model
                ↓
     Probability prediction
                ↓
  0.25 classification threshold
                ↓
Threshold-aware response and disclaimer
```

The LLM does not make the diabetes prediction itself.

Its job is only to convert the natural-language description into structured data.

The trained machine learning model is responsible for the prediction.

---

## Example Prediction

A prediction can produce output similar to:

```text
The model classifies this case as below the diabetes prediction threshold.

Estimated model probability: 23.8%.

Classification threshold: 25.0%.

The estimated probability is very close to the model's classification threshold.

The prediction is 1.2 percentage points from the decision threshold.

This prediction is for educational purposes and is not a medical diagnosis.
Please consult a healthcare professional for personalized medical advice.
```

I intentionally describe results relative to the model threshold rather than using labels such as "low risk" or "high risk."

The model was not trained or calibrated to provide clinical risk categories.

---

## Handling Incomplete Input

The LLM is instructed not to guess missing medical information.

If a required value cannot be determined, it returns `null`.

The application then validates the extracted information and identifies any missing required features.

The interface also validates values such as:

- Gender
- Smoking history
- Hypertension
- Heart disease
- Age
- BMI
- HbA1c
- Blood glucose

This helps prevent invalid data from reaching the prediction model.

---

## Testing

The project includes automated tests using `pytest`.

Tests cover three major areas.

### Preprocessing Tests

The preprocessing tests check things such as:

- Required columns
- Missing columns
- Empty DataFrames
- Missing-value handling
- Median imputation
- Categorical imputation
- Categorical encoding
- Numeric scaling
- Data quality checks
- Real dataset validation

### Model Tests

The model tests confirm that:

- Predictions have the correct shape
- Predictions contain valid binary classes
- The model meets a minimum expected performance level

### Interface Tests

The interface tests check:

- Valid structured patient input
- Missing features
- Invalid binary features
- DataFrame conversion
- Natural-language parsing
- Missing information from the LLM
- Invalid JSON returned by the LLM

The OpenRouter API is mocked during tests.

This means automated testing does not require a real API request or API key.

Run the tests with:

```bash
python -m pytest tests/ -v
```

---

## Continuous Integration

GitHub Actions is used to automatically run the project pipeline after changes are pushed to the `master` branch.

The workflow:

1. Checks out the repository
2. Installs Python 3.12
3. Installs project dependencies
4. Authenticates with AWS
5. Pulls DVC artifacts from S3
6. Runs the automated tests
7. Runs the training pipeline
8. Uploads training metrics
9. Uploads the trained model as a GitHub Actions artifact

The current GitHub Actions workflow passes successfully.

---

## DVC and AWS S3

The dataset and final selected model are managed using DVC.

The actual large files are stored in an AWS S3 DVC remote.

Git stores the small `.dvc` pointer files.

For example:

```text
data/diabetes_prediction_dataset.csv.dvc
models/best_model.pkl.dvc
```

This keeps large datasets and model artifacts out of normal Git history while still making the project reproducible.

After cloning the repository, the DVC artifacts can be restored with:

```bash
dvc pull
```

The selected model metadata is stored in:

```text
metrics/best_model_selection.json
```

This file contains information such as:

- MLflow run ID
- Selected model
- Classification threshold
- Selection metric
- Validation metrics

---

## Project Structure

```text
diabetes_prediction_mlops/
│
├── .dvc/
│
├── .github/
│   └── workflows/
│       └── diabetes-prediction.yml
│
├── configs/
│   └── config.yml
│
├── data/
│   ├── .gitignore
│   └── diabetes_prediction_dataset.csv.dvc
│
├── metrics/
│   └── best_model_selection.json
│
├── models/
│   ├── .gitignore
│   └── best_model.pkl.dvc
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── evaluate.py
│   ├── preprocess.py
│   ├── sweep.py
│   └── train.py
│
├── tests/
│   ├── test_interface.py
│   ├── test_model.py
│   └── test_preprocessing.py
│
├── .env.example
├── .gitignore
├── Dockerfile
├── pytest.ini
├── README.md
└── requirements.txt
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/dylan-lheureux-ts/diabetes_prediction_mlops.git
cd diabetes_prediction_mlops
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Pull the DVC artifacts

AWS credentials with access to the project's S3 bucket are required.

Then run:

```bash
dvc pull
```

This restores the dataset and selected model.

### 5. Configure OpenRouter

Create a `.env` file in the project root.

Example:

```text
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL=openrouter/free
```

The real `.env` file is ignored by Git and should never be committed.

A safe template is included in:

```text
.env.example
```

---

## Running the Application

Run:

```bash
python src/app.py
```

The current CLI example sends a natural-language patient description through the complete pipeline:

```text
Natural language
→ LLM extraction
→ validation
→ machine learning prediction
→ formatted response
```

---

## Running a Training Experiment

To run the model configuration stored in:

```text
configs/config.yml
```

use:

```bash
python src/train.py
```

---

## Running the Experiment Sweep

To run the different model configurations and compare them through MLflow:

```bash
python src/sweep.py
```

The sweep trains the configured experiments, logs them to MLflow, compares completed runs, and selects the best model using validation Average Precision.

The held-out test set is not used during this model-selection process.

---

## Viewing MLflow Experiments

Start the MLflow interface with:

```bash
mlflow ui
```

Then open the local MLflow interface in a browser.

The experiment is named:

```text
diabetes-prediction
```

---

## Configuration

Model settings are stored in:

```text
configs/config.yml
```

This includes:

- Model type
- Train/test split
- Validation split
- Random state
- Missing-value strategy
- Feature scaling
- Logistic Regression parameters
- Random Forest parameters
- XGBoost parameters
- Threshold search range
- F-beta value
- Minimum model performance requirements

Keeping these settings outside the Python code makes experiments easier to reproduce and modify.

---

## Technologies Used

This project uses:

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- MLflow
- DVC
- AWS S3
- OpenRouter
- OpenAI Python client
- Pytest
- Git
- GitHub
- GitHub Actions
- YAML
- Python dotenv

---

## Limitations

There are several important limitations to this project.

The model was trained on a specific public dataset, so its predictions depend on the quality and representativeness of that dataset.

The probability produced by the model should not automatically be interpreted as a patient's true clinical probability of developing or having diabetes.

The `0.25` threshold is a machine learning classification threshold selected from validation performance. It is not a medically established cutoff.

The LLM is only used for structured information extraction, but LLM output can still occasionally be incorrect or malformed. For this reason, extracted values are validated before they are passed to the model.

The application is intended for educational and portfolio purposes and is not a medical diagnostic system.

---

## Reflection

One of the biggest things I learned from this project was that building a machine learning model is only one part of building a complete machine learning application.

It was fairly easy to focus only on finding the model with the highest score, but this project forced me to think more about how the entire system works together.

I had to think about data leakage, train/validation/test separation, threshold selection, experiment tracking, reproducibility, testing, model versioning, CI/CD, and how a user would actually interact with the model.

The imbalanced target was also an important part of the project. Accuracy looked strong, but I learned that accuracy by itself can be misleading when one class is much more common than the other. That is why I spent more time looking at recall, precision, F1, F-beta, AUC-ROC, and Average Precision.

Adding the LLM interface was another useful part of the project because it showed me how a traditional machine learning model can work together with an LLM without asking the LLM to perform the actual prediction. The LLM handles natural-language understanding while the trained model handles the classification.

I also gained more experience working with MLflow, DVC, AWS S3, GitHub Actions, and automated testing. A lot of the challenges were not necessarily caused by the model itself. They came from making all the different parts of the project work together reliably.

If I continued developing this project, I would improve the user interface, add stronger handling for incomplete or out-of-scope questions, experiment with model calibration, add more explainability around individual predictions, and potentially deploy the application as a web service.

Overall, this project helped me better understand what it takes to move from simply training a machine learning model to building a reproducible application around it.

---

## Disclaimer

This project was created for educational and portfolio purposes.

It is not a medical device and should not be used to diagnose diabetes or make healthcare decisions.

Always consult a qualified healthcare professional for medical guidance.
