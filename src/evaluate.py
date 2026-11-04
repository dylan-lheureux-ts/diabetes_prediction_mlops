import json
import pickle
from pathlib import Path

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
)

from sklearn.model_selection import (
    train_test_split,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

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

FINAL_RESULTS_PATH = (
    PROJECT_ROOT
    / "metrics"
    / "final_test_results.json"
)


# ============================================================
# Prediction evaluation
# ============================================================

def evaluate_predictions(
    y_true,
    probabilities,
    threshold=0.5,
    beta=1.6,
):
    """
    Evaluate binary predictions using a specified
    probability threshold.

    Parameters
    ----------
    y_true : array-like
        True binary class labels.

    probabilities : array-like
        Predicted probabilities for the positive class.

    threshold : float, default=0.5
        Probability cutoff used to convert probabilities
        into binary predictions.

    Returns
    -------
    dict
        Accuracy, precision, recall, Fbeta score, ROC-AUC, and PR-AUC.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "accuracy": round(
            float(
                accuracy_score(
                    y_true,
                    predictions,
                )
            ),
            4,
        ),

        "precision": round(
            float(
                precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),
            4,
        ),

        "recall": round(
            float(
                recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),
            4,
        ),

        "f1_score": round(
            float(
                f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),
            4,
        ),

        "fbeta_score": round(
            float(
                fbeta_score(
                    y_true,
                    predictions,
                    beta=beta,
                    zero_division=0,
                )
            ),
            4,
        ),

        "auc_roc": round(
            float(
                roc_auc_score(
                    y_true,
                    probabilities,
                )
            ),
            4,
        ),
        "auc_pr": round(
            float(
                average_precision_score(
                    y_true,
                    probabilities,
                )
            ),
            4,
        ),
    }


# ============================================================
# Validation threshold search
# ============================================================

def find_best_threshold(
    y_true,
    probabilities,
    threshold_min=0.10,
    threshold_max=0.80,
    threshold_step=0.01,
    beta=1.6,
):
    """
    Find the probability threshold that produces
    the highest Fbeta score.

    This function should be used on validation data,
    never on the final held-out test set.
    """

    best_threshold = 0.5

    best_fbeta_score = -1.0

    best_metrics = None

    thresholds = np.arange(
        threshold_min,
        threshold_max + threshold_step,
        threshold_step,
    )

    for threshold in thresholds:

        metrics = evaluate_predictions(
            y_true,
            probabilities,
            threshold=threshold,
            beta=beta,
        )

        if (
            metrics["fbeta_score"]
            > best_fbeta_score
        ):
            best_fbeta_score = (
                metrics["fbeta_score"]
            )

            best_threshold = float(
                round(
                    threshold,
                    2,
                )
            )

            best_metrics = metrics

    return (
        best_threshold,
        best_metrics,
    )


# ============================================================
# Evaluate fitted model at selected threshold
# ============================================================

def evaluate_model_at_threshold(
    model,
    X,
    y,
    threshold,
    beta=1.6,
):
    """
    Evaluate a fitted binary classifier using a
    specified classification threshold.
    """

    probabilities = (
        model.predict_proba(X)[:, 1]
    )

    return evaluate_predictions(
        y,
        probabilities,
        threshold=threshold,
        beta=beta,
    )


# ============================================================
# Final held-out test evaluation
# ============================================================

def run_final_evaluation():
    """
    Evaluate the selected best model exactly once on
    the untouched held-out test set.

    Run this only after model development and
    validation-based model selection are complete.
    """

    # Import here to avoid circular imports because
    # train.py imports evaluation utilities.
    if __package__:
        from .train import (
            config,
            load_data,
        )
    else:
        from train import (
            config,
            load_data,
        )

    # --------------------------------------------------------
    # Verify required artifacts
    # --------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Best model was not found at "
            f"{MODEL_PATH}. "
            "Run sweep.py first."
        )

    if not SELECTION_PATH.exists():
        raise FileNotFoundError(
            "Best-model metadata was not found at "
            f"{SELECTION_PATH}. "
            "Run sweep.py first."
        )

    # --------------------------------------------------------
    # Load selected model
    # --------------------------------------------------------

    with open(
        MODEL_PATH,
        "rb",
    ) as model_file:
        best_model = pickle.load(
            model_file
        )

    # --------------------------------------------------------
    # Load selected threshold and run information
    # --------------------------------------------------------

    with open(
        SELECTION_PATH,
        "r",
        encoding="utf-8",
    ) as metadata_file:
        selection = json.load(
            metadata_file
        )

    best_threshold = float(
        selection["best_threshold"]
    )

    # --------------------------------------------------------
    # Recreate original held-out test split
    # --------------------------------------------------------

    X, y, _ = load_data(
        config
    )

    (
        X_train_val,
        X_test,
        y_train_val,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y,
    )

    # X_train_val and y_train_val are deliberately
    # unused here.
    #
    # We only recreate the deterministic held-out test
    # split that was reserved during training.

    # --------------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------------

    final_metrics = (
        evaluate_model_at_threshold(
            model=best_model,
            X=X_test,
            y=y_test,
            threshold=best_threshold,
            beta=config["fbeta_beta"],
        )
    )

    # --------------------------------------------------------
    # Save final results
    # --------------------------------------------------------

    final_results = {
        "run_id": selection[
            "run_id"
        ],

        "model_type": selection[
            "model_type"
        ],

        "threshold": (
            best_threshold
        ),

        "fbeta_beta": (
            config["fbeta_beta"]
        ),

        "test_size": len(
            X_test
        ),

        "test_metrics": (
            final_metrics
        ),
    }

    FINAL_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        FINAL_RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as results_file:
        json.dump(
            final_results,
            results_file,
            indent=2,
        )

    # --------------------------------------------------------
    # Display final results
    # --------------------------------------------------------

    print(
        f"\n{'=' * 70}"
    )

    print(
        "FINAL HELD-OUT TEST RESULTS"
    )

    print(
        f"{'=' * 70}"
    )

    print(
        f"MLflow Run ID: "
        f"{selection['run_id']}"
    )

    print(
        f"Model: "
        f"{selection['model_type']}"
    )

    print(
        f"Threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Test Samples: "
        f"{len(X_test):,}"
    )

    print()

    print(
        f"Accuracy:  "
        f"{final_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{final_metrics['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{final_metrics['recall']:.4f}"
    )

    print(
        f"F1 Score:  "
        f"{final_metrics['f1_score']:.4f}"
    )

    print(
        f"Fbeta Score:  "
        f"{final_metrics['fbeta_score']:.4f}"
    )

    print(
        f"AUC-ROC:   "
        f"{final_metrics['auc_roc']:.4f}"
    )

    print(
        f"AUC-PR:    "
        f"{final_metrics['auc_pr']:.4f}"
    )

    print(
        f"{'=' * 70}"
    )

    print(
        f"\nFinal results saved to: "
        f"{FINAL_RESULTS_PATH}"
    )

    return final_results


# ============================================================
# Run final evaluation
# ============================================================

if __name__ == "__main__":
    run_final_evaluation()