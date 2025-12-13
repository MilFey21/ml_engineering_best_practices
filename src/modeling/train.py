from pathlib import Path

import joblib
from loguru import logger
import mlflow
import mlflow.sklearn
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
    max_depth: int = 8,
    experiment_name: str = "churn_prediction",
    run_name: str = None,
):
    """Train a baseline Random Forest model for customer churn prediction."""
    # Set MLflow tracking URI (defaults to local ./mlruns)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)

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

    # Start MLflow run
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("features_path", str(features_path))
        mlflow.log_param("labels_path", str(labels_path))

        # Log data info
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_samples", X.shape[0])
        mlflow.log_param("n_train_samples", X_train.shape[0])
        mlflow.log_param("n_test_samples", X_test.shape[0])

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

        # Log metrics to MLflow
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1_score", test_f1)
        mlflow.log_metric("test_roc_auc", test_roc_auc)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)

        # Feature importance
        feature_importance = pd.DataFrame(
            {
                "feature": X.columns,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        # Log model to MLflow
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="RandomForestChurn",
        )

        # Log feature importance as artifact
        importance_path = MODELS_DIR / "feature_importance.csv"
        feature_importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(str(importance_path), "feature_importance")

        # Log confusion matrix
        cm_df = pd.DataFrame(
            cm,
            index=["No Churn", "Churn"],
            columns=["No Churn", "Churn"],
        )
        cm_path = MODELS_DIR / "confusion_matrix.csv"
        cm_df.to_csv(cm_path)
        mlflow.log_artifact(str(cm_path), "metrics")

        # Save the model locally as well
        joblib.dump(model, model_path)

        # Log model path
        mlflow.log_param("local_model_path", str(model_path))

        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run ID: {run_id}")

    # Log metrics to console
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

    logger.info(f"Confusion Matrix:\n{cm}")

    # Classification report
    logger.info("\nClassification Report:")
    logger.info(f"\n{classification_report(y_test, y_test_pred)}")

    logger.info("\nTop 10 Most Important Features:")
    logger.info(f"\n{feature_importance.head(10).to_string(index=False)}")

    logger.success(f"Model saved to {model_path}")
    logger.info(f"Feature importance saved to {importance_path}")


if __name__ == "__main__":
    app()
