"""Reusable model evaluation utilities."""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def evaluate_model(
    model,
    X_test,
    y_test,
    metric_average="binary",
    train_size=None,
    n_features=None,
):
    """Evaluate a fitted classification model and return its metrics.

    Parameters
    ----------
    model : estimator
        A fitted scikit-learn classification model with ``predict`` and
        ``predict_proba`` methods.
    X_test : pandas.DataFrame or array-like
        Test features.
    y_test : pandas.Series or array-like
        Test target values.
    metric_average : str, default="binary"
        Averaging method used for precision, recall, and F1.
    train_size : int, optional
        Number of training samples, included in the returned metrics when
        provided.
    n_features : int, optional
        Number of model features, included in the returned metrics when
        provided.

    Returns
    -------
    dict
        Accuracy, precision, recall, F1 score, ROC-AUC, and optional dataset
        size information.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    class_count = len(model.classes_)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average=metric_average, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, average=metric_average, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, average=metric_average, zero_division=0
    )

    if class_count == 2:
        auc = roc_auc_score(y_test, y_prob[:, 1])
    else:
        auc = roc_auc_score(
            y_test,
            y_prob,
            multi_class="ovr",
            average=metric_average,
        )

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "auc_roc": round(auc, 4),
        "test_size": len(X_test),
    }

    if train_size is not None:
        metrics["train_size"] = train_size
    if n_features is not None:
        metrics["n_features"] = n_features

    return metrics