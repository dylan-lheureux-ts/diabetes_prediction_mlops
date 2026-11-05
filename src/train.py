import json
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
if __package__:
    from .evaluate import (
        evaluate_predictions,
        find_best_threshold,
    )
else:
    from evaluate import (
        evaluate_predictions,
        find_best_threshold,
    )

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


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

    if model_type == "xgboost":

        return XGBClassifier(
            n_estimators=config["xgb_n_estimators"],
            learning_rate=config["xgb_learning_rate"],
            max_depth=config["xgb_max_depth"],
            subsample=config["xgb_subsample"],
            colsample_bytree=config["xgb_colsample_bytree"],
            objective="binary:logistic",
            random_state=config["random_state"],
            n_jobs=-1,
        )
    raise ValueError(
        f"Unknown model type: {model_type}"
    )


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
        # Train / validation / test split
        # ----------------------------------------------------

        X_train_val, X_test, y_train_val, y_test = (
            train_test_split(
                X,
                y,
                test_size=config["test_size"],
                random_state=config["random_state"],
                stratify=y,
            )
        )

        X_train, X_val, y_train, y_val = (
         train_test_split(
                X_train_val,
                y_train_val,
                test_size=config["validation_size"],
                random_state=config["random_state"],
                stratify=y_train_val,
            )
        )

        print(
            f"Training rows: {len(X_train):,}"
        )

        print(
            f"Validation rows: {len(X_val):,}"
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

        mlflow.log_param(
            "validation_size",
            config["validation_size"],
        )

        mlflow.log_param(
            "threshold_min",
            config["threshold_min"],
        )

        mlflow.log_param(
            "threshold_max",
            config["threshold_max"],
        )

        mlflow.log_param(
            "threshold_step",
            config["threshold_step"],
        )

        mlflow.log_param(
            "fbeta_beta",
            config["fbeta_beta"],
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

        elif model_type == "xgboost":

            mlflow.log_param(
                "n_estimators",
                config["xgb_n_estimators"],
            )

            mlflow.log_param(
                "learning_rate",
                config["xgb_learning_rate"],
            )

            mlflow.log_param(
                "max_depth",
                config["xgb_max_depth"],
            )
            mlflow.log_param(
                "subsample",
                config["xgb_subsample"],
            )
            mlflow.log_param(
                "colsample_bytree",
                config["xgb_colsample_bytree"],
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

        validation_probabilities = (
            pipeline.predict_proba(X_val)[:, 1]
        )

        best_threshold, validation_metrics = (
            find_best_threshold(
                y_val,
                validation_probabilities,
                threshold_min=config[
                    "threshold_min"
                ],
                threshold_max=config[
                    "threshold_max"
                ],
                threshold_step=config[
                    "threshold_step"
                ],
                beta=config[
                    "fbeta_beta"
                ],
            )
        )

        print(
            f"\nBest validation threshold: "
            f"{best_threshold:.2f}"
        )

        print(
            f"Validation F1: "
            f"{validation_metrics['f1_score']:.4f}"
        )

        print(
            f"Validation Fbeta: "
            f"{validation_metrics['fbeta_score']:.4f}"
        )

        print(
            f"Validation Precision: "
            f"{validation_metrics['precision']:.4f}"
        )

        print(
            f"Validation Recall: "
            f"{validation_metrics['recall']:.4f}"
        )

        print(
            f"Validation AUC-ROC: "
            f"{validation_metrics['auc_roc']:.4f}"
        )

        print(
            f"Validation Avg Precision: "
            f"{validation_metrics['avg_precision']:.4f}"
        )

        # ----------------------------------------------------
        # Log validation metrics to MLflow
        # ----------------------------------------------------

        mlflow.log_metric(
            "validation_accuracy",
            validation_metrics["accuracy"],
        )

        mlflow.log_metric(
            "validation_precision",
            validation_metrics["precision"],
        )

        mlflow.log_metric(
            "validation_recall",
            validation_metrics["recall"],
        )

        mlflow.log_metric(
            "validation_f1",
            validation_metrics["f1_score"],
        )

        mlflow.log_metric(
            "validation_fbeta",
            validation_metrics["fbeta_score"],
        )

        mlflow.log_metric(
            "validation_auc_roc",
            validation_metrics["auc_roc"],
        )

        mlflow.log_metric(
            "validation_avg_precision",
            validation_metrics["avg_precision"],
        )

        mlflow.log_param(
            "best_threshold",
            best_threshold,
        )

        mlflow.log_param(
            "fbeta_beta",
            config["fbeta_beta"],
        )

        # ----------------------------------------------------
        # Performance thresholds
        # ----------------------------------------------------

        min_accuracy = config.get(
            "min_accuracy",
            0.0,
        )

        min_fbeta = config.get(
            "min_fbeta",
            0.0,
        )

        if (
            validation_metrics["accuracy"]
            < min_accuracy
        ):

            print(
                "\nWARNING: Accuracy "
                f"{validation_metrics['accuracy']} "
                "is below threshold "
                f"{min_accuracy}"
            )

        if (
            validation_metrics["fbeta_score"]
            < min_fbeta
        ):

            print(
                "\nWARNING: Fbeta score "
                f"{validation_metrics['fbeta_score']} "
                "is below threshold "
                f"{min_fbeta}"
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
        run_metrics = {
            **validation_metrics,
            "best_threshold": best_threshold,
            "train_size": len(X_train),
            "validation_size": len(X_val),
            "test_size": len(X_test),
            "n_features": X_train.shape[1],
        }
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
                run_metrics,
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
            name="model",
            serialization_format="cloudpickle",
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
            f"Best Threshold:     "
            f"{best_threshold:.2f}"
        )

        print(
            f"Validation Accuracy:  "
            f"{validation_metrics['accuracy']:.4f}"
        )

        print(
            f"Validation Precision: "
            f"{validation_metrics['precision']:.4f}"
        )

        print(
            f"Validation Recall:    "
            f"{validation_metrics['recall']:.4f}"
        )

        print(
            f"Validation F1 Score:  "
            f"{validation_metrics['f1_score']:.4f}"
        )

        print(
            f"Validation Fbeta Score:  "
            f"{validation_metrics['fbeta_score']:.4f}"
        )

        print(
            f"Validation AUC-ROC:   "
            f"{validation_metrics['auc_roc']:.4f}"
        )

        print(
            f"Validation Avg Precision:   "
            f"{validation_metrics['avg_precision']:.4f}"
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
            validation_metrics["accuracy"]
            < min_accuracy
        ):

            print(
                "\nFAILED: Accuracy "
                "below threshold"
            )

            sys.exit(1)

        if (
            validation_metrics["fbeta_score"]
            < min_fbeta
        ):

            print(
                "\nFAILED: Fbeta score "
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
