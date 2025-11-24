from pathlib import Path

import joblib
from loguru import logger
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import typer

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

MODELS_DIR.mkdir(parents=True, exist_ok=True)


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: int = 10,
):
    """Train a baseline Random Forest model for customer churn prediction."""
    logger.info("Loading features and labels...")

    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path).squeeze()

    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Labels shape: {y.shape}")
    logger.info(f"Class distribution:\n{y.value_counts()}")

    # Split data into train and test sets
    logger.info(f"Splitting data into train/test sets (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")

    # Initialize and train the model
    logger.info("Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",  # Handle class imbalance
    )

    model.fit(X_train, y_train)

    # Make predictions
    logger.info("Making predictions...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)

    # Log metrics
    logger.info("=" * 50)
    logger.info("Model Performance Metrics:")
    logger.info("=" * 50)
    logger.info(f"Train Accuracy: {train_accuracy:.4f}")
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Test Precision: {test_precision:.4f}")
    logger.info(f"Test Recall: {test_recall:.4f}")
    logger.info(f"Test F1-Score: {test_f1:.4f}")
    logger.info(f"Test ROC-AUC: {test_roc_auc:.4f}")
    logger.info("=" * 50)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    logger.info(f"Confusion Matrix:\n{cm}")

    # Classification report
    logger.info("\nClassification Report:")
    logger.info(f"\n{classification_report(y_test, y_test_pred)}")

    # Feature importance
    feature_importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    logger.info("\nTop 10 Most Important Features:")
    logger.info(f"\n{feature_importance.head(10).to_string(index=False)}")

    # Save the model
    joblib.dump(model, model_path)
    logger.success(f"Model saved to {model_path}")

    # Save feature importance
    importance_path = MODELS_DIR / "feature_importance.csv"
    feature_importance.to_csv(importance_path, index=False)
    logger.info(f"Feature importance saved to {importance_path}")


if __name__ == "__main__":
    app()
