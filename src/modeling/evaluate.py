"""Model evaluation script."""

import json
from pathlib import Path

import joblib
from loguru import logger
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import typer

app = typer.Typer()


@app.command()
def main(
    model_path: Path,
    features_path: Path,
    labels_path: Path,
    output_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Evaluate a trained model."""
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    logger.info(f"Loading features from {features_path}")
    logger.info(f"Loading labels from {labels_path}")

    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path).squeeze()

    # Split data (same split as training)
    from sklearn.model_selection import train_test_split

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Test set: {X_test.shape[0]} samples")

    # Make predictions
    logger.info("Making predictions...")
    y_pred = model.predict(X_test)

    # Calculate probabilities for ROC-AUC
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    except (AttributeError, IndexError):
        y_proba = None
        roc_auc = None

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Prepare evaluation report
    evaluation_report = {
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        },
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    if roc_auc is not None:
        evaluation_report["metrics"]["roc_auc"] = float(roc_auc)

    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(evaluation_report, f, indent=2)

    logger.info("Evaluation Report:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-Score: {f1:.4f}")
    if roc_auc is not None:
        logger.info(f"  ROC-AUC: {roc_auc:.4f}")

    logger.info(f"Report saved to {output_path}")


if __name__ == "__main__":
    app()
