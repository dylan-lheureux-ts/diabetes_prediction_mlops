import json
from pathlib import Path

import cloudpickle
import mlflow
import mlflow.sklearn


# Support both:
# python src/sweep.py
# python -m src.sweep
if __package__:
    from .train import (
        config as base_config,
        run_experiment,
    )
else:
    from train import (
        config as base_config,
        run_experiment,
    )


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"

METRICS_DIR = PROJECT_ROOT / "metrics"


# ============================================================
# Experiment configurations
# ============================================================

experiments = [
    # Logistic Regression
    {
        "model_type": "logistic_regression",
        "lr_C": 0.1,
    },
    {
        "model_type": "logistic_regression",
        "lr_C": 1.0,
    },

    # Random Forest
    {
        "model_type": "random_forest",
        "rf_n_estimators": 500,
        "rf_max_depth": 12,
    },
    {
        "model_type": "random_forest",
        "rf_n_estimators": 400,
        "rf_max_depth": 7,
    },

    # XGBoost
    {
        "model_type": "xgboost",
        "xgb_n_estimators": 400,
        "xgb_learning_rate": 0.04,
        "xgb_max_depth": 3,
        "xgb_subsample": 0.8,
        "xgb_colsample_bytree": 0.8,
    },
    {
        "model_type": "xgboost",
        "xgb_n_estimators": 300,
        "xgb_learning_rate": 0.01,
        "xgb_max_depth": 5,
        "xgb_subsample": 0.8,
        "xgb_colsample_bytree": 0.8,
    }
]

# ============================================================
# Run experiments
# ============================================================


def run_sweep():
    """
    Run all model configurations.

    Each experiment:
    - trains on the training set
    - tunes threshold on the validation set
    - logs validation metrics to MLflow

    The held-out test set is NOT evaluated here.
    """

    print(
        f"\nRunning {len(experiments)} experiments..."
    )

    successful_runs = []

    for i, overrides in enumerate(
        experiments,
        start=1,
    ):
        print(
            f"\n{'=' * 60}"
        )

        print(
            f"Experiment {i}/{len(experiments)}"
        )

        print(
            f"{'=' * 60}"
        )

        current_config = base_config.copy()
        current_config.update(
            overrides
        )

        print(
            f"Model: "
            f"{current_config['model_type']}"
        )

        try:
            run_id = run_experiment(
                current_config
            )

            successful_runs.append(
                run_id
            )

            print(
                "\nCompleted successfully."
            )

            print(
                f"Run ID: {run_id}"
            )

        except SystemExit as exc:
            print(
                "\nExperiment failed a "
                "performance threshold."
            )

            print(
                f"Exit code: {exc.code}"
            )

        except Exception as exc:
            print(
                f"\nExperiment failed: {exc}"
            )

    return successful_runs


# ============================================================
# Compare MLflow experiments
# ============================================================

def compare_runs():
    """
    Compare completed MLflow runs using validation AUC-PR score.

    The best run is selected using validation performance only.

    The held-out test set is NOT accessed here.
    """

    experiment = mlflow.get_experiment_by_name(
        "diabetes-prediction"
    )

    if experiment is None:
        raise ValueError(
            "MLflow experiment "
            "'diabetes-prediction' was not found."
        )

    # --------------------------------------------------------
    # Search completed runs
    # --------------------------------------------------------

    runs = mlflow.search_runs(
        experiment_ids=[
            experiment.experiment_id
        ],
        filter_string=(
            "attributes.status = 'FINISHED'"
        ),
        order_by=[
            "metrics.validation_auc_pr DESC"
        ],
    )

    # Remove older runs from before validation metrics
    # were introduced.
    runs = runs.dropna(
        subset=[
            "metrics.validation_auc_pr",
        ]
    )

    if runs.empty:
        raise ValueError(
            "No completed MLflow runs with "
            "validation metrics were found."
        )

    # --------------------------------------------------------
    # Display validation comparison
    # --------------------------------------------------------

    comparison_columns = [
        "run_id",
        "params.model_type",
        "params.best_threshold",
        "params.fbeta_beta",
        "metrics.validation_accuracy",
        "metrics.validation_precision",
        "metrics.validation_recall",
        "metrics.validation_f1",
        "metrics.validation_fbeta",
        "metrics.validation_auc_roc",
        "metrics.validation_auc_pr",
    ]

    available_columns = [
        column
        for column in comparison_columns
        if column in runs.columns
    ]

    comparison = runs[
        available_columns
    ].copy()

    print(
        f"\n{'=' * 90}"
    )

    print(
        "MLFLOW VALIDATION COMPARISON"
    )

    print(
        f"{'=' * 90}"
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Select best run
    # --------------------------------------------------------

    best_run = runs.iloc[0]

    best_run_id = best_run[
        "run_id"
    ]

    best_model_type = best_run[
        "params.model_type"
    ]

    best_threshold = float(
        best_run[
            "params.best_threshold"
        ]
    )

    # --------------------------------------------------------
    # Load exact model artifact from winning run
    # --------------------------------------------------------

    best_model_uri = (
        f"runs:/{best_run_id}/model"
    )

    best_model = mlflow.sklearn.load_model(
        best_model_uri
    )

    # --------------------------------------------------------
    # Save best model locally
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_model_path = (
        MODEL_DIR
        / "best_model.pkl"
    )

    with open(
        best_model_path,
        "wb",
    ) as model_file:
        cloudpickle.dump(
            best_model,
            model_file,
        )

    # --------------------------------------------------------
    # Save model-selection metadata
    # --------------------------------------------------------

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    selection_path = (
        METRICS_DIR
        / "best_model_selection.json"
    )

    selection_metadata = {
        "run_id": best_run_id,
        "model_type": best_model_type,
        "best_threshold": best_threshold,
        "selection_metric": (
            "validation_auc_pr"
        ),
        "validation_metrics": {
            "accuracy": float(
                best_run[
                    "metrics.validation_accuracy"
                ]
            ),
            "precision": float(
                best_run[
                    "metrics.validation_precision"
                ]
            ),
            "recall": float(
                best_run[
                    "metrics.validation_recall"
                ]
            ),
            "f1_score": float(
                best_run[
                    "metrics.validation_f1"
                ]
            ),
            "fbeta_score": float(
                best_run[
                    "metrics.validation_fbeta"
                ]
            ),
            "auc_roc": float(
                best_run[
                    "metrics.validation_auc_roc"
                ]
            ),
            "auc_pr": float(
                best_run[
                    "metrics.validation_auc_pr"
                ]
            ),
        },
    }

    with open(
        selection_path,
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            selection_metadata,
            metadata_file,
            indent=2,
        )

    # --------------------------------------------------------
    # Display selected run
    # --------------------------------------------------------

    print(
        f"\n{'=' * 70}"
    )

    print(
        "SELECTED MODEL"
    )

    print(
        f"{'=' * 70}"
    )

    print(
        f"Run ID: "
        f"{best_run_id}"
    )

    print(
        f"Model: "
        f"{best_model_type}"
    )

    print(
        f"Best Threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Validation Accuracy: "
        f"{best_run['metrics.validation_accuracy']:.4f}"
    )

    print(
        f"Validation Precision: "
        f"{best_run['metrics.validation_precision']:.4f}"
    )

    print(
        f"Validation Recall: "
        f"{best_run['metrics.validation_recall']:.4f}"
    )

    print(
        f"Validation F1: "
        f"{best_run['metrics.validation_f1']:.4f}"
    )

    print(
        f"Validation Fbeta: "
        f"{best_run['metrics.validation_fbeta']:.4f}"
    )

    print(
        f"Validation AUC-ROC: "
        f"{best_run['metrics.validation_auc_roc']:.4f}"
    )

    print(
        f"Validation AUC-PR: "
        f"{best_run['metrics.validation_auc_pr']:.4f}"
    )

    print(
        "\nSelection criterion: "
        "highest validation AUC-PR score."
    )

    print(
        "\nIMPORTANT: The held-out test set "
        "has NOT been evaluated."
    )

    print(
        f"\nBest model saved to: "
        f"{best_model_path}"
    )

    print(
        f"Selection metadata saved to: "
        f"{selection_path}"
    )

    return selection_metadata


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    successful_runs = run_sweep()

    print(
        f"\n{len(successful_runs)} "
        "experiments completed successfully."
    )

    compare_runs()

    print(
        "\nExperiment sweep complete."
    )

    print(
        "The held-out test set remains untouched."
    )

    print(
        "Run final evaluation only when "
        "model development is complete."
    )

    print(
        "Run 'mlflow ui' to inspect all runs."
    )
