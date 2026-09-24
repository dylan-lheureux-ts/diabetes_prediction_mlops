import json
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yml"

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "diabetes_prediction_dataset.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "metrics"


# ============================================================
# Dataset configuration
# ============================================================


CATEGORICAL_COLUMNS = [
    "gender",
    "smoking_history",
]

NUMERIC_COLUMNS = [
    "age",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]

BINARY_COLUMNS = [
    "hypertension",
    "heart_disease",
]


# ============================================================
# Configuration
# ============================================================

def load_config(config_path=CONFIG_PATH):
    """Load experiment settings from the YAML configuration file."""

    with open(config_path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration must contain a YAML mapping: {config_path}"
        )

    config.pop("name", None)

    return config


config = load_config()


# ============================================================
# Load data
# ============================================================

def load_data(config):
    """Load and validate the diabetes prediction dataset."""

    df = pd.read_csv(DATA_PATH)

    print(
        f"Loaded {len(df):,} rows and "
        f"{len(df.columns)} columns."
    )

    # Validate required columns
    target_column = config["target_column"]

    required_columns = (
        CATEGORICAL_COLUMNS
        + NUMERIC_COLUMNS
        + BINARY_COLUMNS
        + [target_column]
    )

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Remove duplicate records
    before = len(df)

    df = df.drop_duplicates().reset_index(drop=True)

    removed = before - len(df)

    print(f"Removed {removed:,} duplicate rows.")
    print(f"Rows remaining: {len(df):,}")

    # Drop optional user-specified features
    features_to_drop = config.get(
        "features_to_drop",
        [],
    )

    if features_to_drop:
        df = df.drop(
            columns=features_to_drop,
            errors="ignore",
        )

        print(
            f"Dropped configured features: "
            f"{features_to_drop}"
        )

    target_column = config["target_column"]

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y, len(df)


# ============================================================
# Preprocessing pipeline
# ============================================================

def build_preprocessor(config):
    """
    Build preprocessing steps for numeric, categorical,
    and binary features.
    """

    numeric_steps = [
        (
            "imputer",
            SimpleImputer(strategy="median"),
        )
    ]

    if config.get("scale_features", True):
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipeline = Pipeline(
        steps=numeric_steps
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    binary_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_COLUMNS,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_COLUMNS,
            ),
            (
                "binary",
                binary_pipeline,
                BINARY_COLUMNS,
            ),
        ]
    )

    return preprocessor


# ============================================================
# Model
# ============================================================

def build_model(config):
    """Create a classification model from config.yml."""

    model_type = config["model_type"]

    if model_type == "logistic_regression":

        return LogisticRegression(
            C=config["lr_C"],
            random_state=config["random_state"],
            max_iter=1000,
            class_weight="balanced",
        )

    if model_type == "random_forest":

        return RandomForestClassifier(
            n_estimators=config["rf_n_estimators"],
            max_depth=config["rf_max_depth"],
            random_state=config["random_state"],
            class_weight="balanced",
        )

    if model_type == "gradient_boosting":

        return GradientBoostingClassifier(
            n_estimators=config["gb_n_estimators"],
            learning_rate=config["gb_learning_rate"],
            max_depth=config["gb_max_depth"],
            random_state=config["random_state"],
        )

    raise ValueError(
        f"Unknown model type: {model_type}"
    )


# ============================================================
# Evaluation
# ============================================================

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate classification evaluation metrics."""

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "auc_roc": roc_auc_score(
            y_true,
            y_prob,
        ),
    }


# ============================================================
# MLflow experiment
# ============================================================

def run_experiment(config):
    """
    Train one model configuration and track the
    experiment using MLflow.
    """

    mlflow.set_experiment(
        "diabetes-prediction"
    )

    with mlflow.start_run():

        # ----------------------------------------------------
        # Load data
        # ----------------------------------------------------

        X, y, n_rows = load_data(config)

        # ----------------------------------------------------
        # Train/test split
        # ----------------------------------------------------

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=config["test_size"],
                random_state=config["random_state"],
                stratify=y,
            )
        )

        print(
            f"Training rows: {len(X_train):,}"
        )

        print(
            f"Testing rows: {len(X_test):,}"
        )

        # ----------------------------------------------------
        # Build complete ML pipeline
        # ----------------------------------------------------

        preprocessor = build_preprocessor(config)

        classifier = build_model(config)

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "classifier",
                    classifier,
                ),
            ]
        )

        # ----------------------------------------------------
        # Log general parameters
        # ----------------------------------------------------

        mlflow.log_param(
            "model_type",
            config["model_type"],
        )

        mlflow.log_param(
            "test_size",
            config["test_size"],
        )

        mlflow.log_param(
            "random_state",
            config["random_state"],
        )

        mlflow.log_param(
            "scale_features",
            config.get(
                "scale_features",
                True,
            ),
        )

        mlflow.log_param(
            "n_rows",
            n_rows,
        )

        mlflow.log_param(
            "dataset",
            "diabetes_prediction_dataset.csv",
        )

        mlflow.log_param(
            "features_dropped",
            str(
                config.get(
                    "features_to_drop",
                    [],
                )
            ),
        )

        # ----------------------------------------------------
        # Log model-specific hyperparameters
        # ----------------------------------------------------

        model_type = config["model_type"]

        if model_type == "logistic_regression":

            mlflow.log_param(
                "C",
                config["lr_C"],
            )

        elif model_type == "random_forest":

            mlflow.log_param(
                "n_estimators",
                config["rf_n_estimators"],
            )

            mlflow.log_param(
                "max_depth",
                config["rf_max_depth"],
            )

        elif model_type == "gradient_boosting":

            mlflow.log_param(
                "n_estimators",
                config["gb_n_estimators"],
            )

            mlflow.log_param(
                "learning_rate",
                config["gb_learning_rate"],
            )

            mlflow.log_param(
                "max_depth",
                config["gb_max_depth"],
            )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        print(
            f"\nTraining "
            f"{config['model_type']}..."
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        y_pred = pipeline.predict(X_test)

        y_prob = pipeline.predict_proba(
            X_test
        )[:, 1]

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics = calculate_metrics(
            y_test,
            y_pred,
            y_prob,
        )

        rounded_metrics = {
            key: round(value, 4)
            for key, value in metrics.items()
        }

        rounded_metrics["train_size"] = len(
            X_train
        )

        rounded_metrics["test_size"] = len(
            X_test
        )

        rounded_metrics["n_features"] = (
            X_train.shape[1]
        )

        # ----------------------------------------------------
        # Log metrics to MLflow
        # ----------------------------------------------------

        for metric_name in [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "auc_roc",
        ]:

            mlflow.log_metric(
                metric_name,
                rounded_metrics[metric_name],
            )

        # ----------------------------------------------------
        # Performance thresholds
        # ----------------------------------------------------

        min_accuracy = config.get(
            "min_accuracy",
            0.0,
        )

        min_f1 = config.get(
            "min_f1",
            0.0,
        )

        if (
            rounded_metrics["accuracy"]
            < min_accuracy
        ):

            print(
                "\nWARNING: Accuracy "
                f"{rounded_metrics['accuracy']} "
                "is below threshold "
                f"{min_accuracy}"
            )

        if (
            rounded_metrics["f1_score"]
            < min_f1
        ):

            print(
                "\nWARNING: F1 score "
                f"{rounded_metrics['f1_score']} "
                "is below threshold "
                f"{min_f1}"
            )

        # ----------------------------------------------------
        # Save model pipeline
        # ----------------------------------------------------

        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_path = (
            MODEL_DIR
            / "model.pkl"
        )

        pd.to_pickle(
            pipeline,
            model_path,
        )

        print(
            f"\nModel pipeline saved to "
            f"{model_path}"
        )

        # ----------------------------------------------------
        # Save metrics
        # ----------------------------------------------------

        METRICS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        metrics_path = (
            METRICS_DIR
            / "results.json"
        )

        with open(
            metrics_path,
            "w",
            encoding="utf-8",
        ) as metrics_file:

            json.dump(
                rounded_metrics,
                metrics_file,
                indent=2,
            )

        print(
            f"Metrics saved to "
            f"{metrics_path}"
        )

        # ----------------------------------------------------
        # MLflow model artifact
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            pipeline,
            "model",
        )

        # ----------------------------------------------------
        # Log config snapshot
        # ----------------------------------------------------

        config_snapshot = (
            PROJECT_ROOT
            / "config_snapshot.json"
        )

        with open(
            config_snapshot,
            "w",
            encoding="utf-8",
        ) as config_file:

            json.dump(
                config,
                config_file,
                indent=2,
                default=str,
            )

        mlflow.log_artifact(
            str(config_snapshot)
        )

        config_snapshot.unlink()

        # ----------------------------------------------------
        # Display results
        # ----------------------------------------------------

        print(
            f"\n{'=' * 50}"
        )

        print(
            f"Model:     "
            f"{config['model_type']}"
        )

        print(
            f"Accuracy:  "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Precision: "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Recall:    "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"F1 Score:  "
            f"{metrics['f1_score']:.4f}"
        )

        print(
            f"AUC-ROC:   "
            f"{metrics['auc_roc']:.4f}"
        )

        print(
            f"{'=' * 50}"
        )

        # ----------------------------------------------------
        # Get MLflow run ID
        # ----------------------------------------------------

        run_id = (
            mlflow.active_run()
            .info
            .run_id
        )

        print(
            f"\nMLflow Run ID: {run_id}"
        )

        print(
            "View experiments with: "
            "mlflow ui"
        )

        # ----------------------------------------------------
        # Fail CI if thresholds are not met
        # ----------------------------------------------------

        if (
            metrics["accuracy"]
            < min_accuracy
        ):

            print(
                "\nFAILED: Accuracy "
                "below threshold"
            )

            sys.exit(1)

        if (
            metrics["f1_score"]
            < min_f1
        ):

            print(
                "\nFAILED: F1 score "
                "below threshold"
            )

            sys.exit(1)

        print(
            "\nAll thresholds passed!"
        )

    return run_id


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    run_experiment(config)