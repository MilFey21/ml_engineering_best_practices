"""Script for running multiple ML experiments with different models and hyperparameters using ClearML."""

from pathlib import Path
from typing import Any, Dict, cast

from clearml import Task
import joblib
from loguru import logger
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.modeling.clearml_utils import ClearMLExperiment, compare_experiments

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def create_model(model_type: str, params: Dict[str, Any], random_state: int = 42):
    """Create a model instance based on model type and parameters."""
    model_configs = {
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "logistic_regression": LogisticRegression,
        "svm": SVC,
        "knn": KNeighborsClassifier,
        "decision_tree": DecisionTreeClassifier,
        "naive_bayes": GaussianNB,
        "adaboost": AdaBoostClassifier,
    }

    if model_type not in model_configs:
        raise ValueError(f"Unknown model type: {model_type}")

    ModelClass = model_configs[model_type]
    model_params = params.copy()

    # Handle models that don't accept random_state
    models_without_random_state = ["knn", "naive_bayes"]
    if model_type not in models_without_random_state:
        model_params["random_state"] = random_state

    # Handle special cases
    if model_type == "svm":
        model_params["probability"] = True  # Required for ROC-AUC

    return ModelClass(**model_params)


def train_and_log_experiment(
    model_type: str,
    model_params: Dict[str, Any],
    features_path: Path,
    labels_path: Path,
    project_name: str = "Churn Prediction",
    task_name: str = None,
    test_size: float = 0.2,
    random_state: int = 42,
    tags: list = None,
):
    """Train a model and log experiment to ClearML using context manager."""
    # Generate task name if not provided
    exp_task_name = task_name or f"{model_type}_{id(model_params)}"

    # Add model type tag
    exp_tags = tags or []
    if model_type not in exp_tags:
        exp_tags.append(model_type)

    # Use context manager for ClearML experiment
    with ClearMLExperiment(
        project_name=project_name,
        task_name=exp_task_name,
        task_type=Task.TaskTypes.training,
        tags=exp_tags,
    ) as exp:
        # Prepare all parameters for logging
        all_params = {
            **model_params,
            "model_type": model_type,
            "test_size": test_size,
            "random_state": random_state,
            "features_path": str(features_path),
            "labels_path": str(labels_path),
        }

        # Load data
        logger.info(f"Loading data for {model_type}...")
        X = pd.read_csv(features_path)
        y = pd.read_csv(labels_path).squeeze()

        # Add data info to parameters
        all_params.update(
            {
                "n_features": X.shape[1],
                "n_samples": X.shape[0],
            }
        )

        # Log hyperparameters
        exp.log_params(all_params)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Log split info
        exp.log_params(
            {
                "n_train_samples": X_train.shape[0],
                "n_test_samples": X_test.shape[0],
            }
        )

        # Create and train model with iteration logging
        logger.info(f"Training {model_type} with params: {model_params}")

        # Check if model supports incremental training (has n_estimators)
        n_iterations = model_params.get("n_estimators", None)
        supports_warm_start = model_type in ["random_forest", "adaboost", "gradient_boosting"]

        if n_iterations and supports_warm_start:
            # Train incrementally and log metrics at each iteration/epoch
            model_params_warm = model_params.copy()
            model_params_warm["warm_start"] = True
            model_params_warm["n_estimators"] = 1

            # Create model with warm_start
            model = create_model(model_type, model_params_warm, random_state=random_state)

            # Determine logging frequency (log every N iterations, but at least 10 times)
            log_frequency = max(1, n_iterations // 10)

            # Train incrementally
            for iteration in range(1, n_iterations + 1):
                model.n_estimators = iteration
                model.fit(X_train, y_train)

                # Calculate metrics at this iteration
                y_train_pred_iter = model.predict(X_train)
                y_test_pred_iter = model.predict(X_test)
                train_acc_iter = accuracy_score(y_train, y_train_pred_iter)
                test_acc_iter = accuracy_score(y_test, y_test_pred_iter)
                test_f1_iter = f1_score(y_test, y_test_pred_iter)

                # Log metrics for this iteration/epoch
                exp.log_metrics(
                    {
                        "train_accuracy": train_acc_iter,
                        "test_accuracy": test_acc_iter,
                        "test_f1_score": test_f1_iter,
                    },
                    iteration=iteration,
                )

                # Log progress periodically
                if iteration % log_frequency == 0 or iteration == n_iterations:
                    logger.info(
                        f"Epoch {iteration}/{n_iterations}: Test F1={test_f1_iter:.4f}, Test Acc={test_acc_iter:.4f}"
                    )
        else:
            # For non-iterative models, train normally (single "epoch")
            model = create_model(model_type, model_params, random_state=random_state)
            model.fit(X_train, y_train)

            # Log metrics at iteration 0 for consistency
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred)

            exp.log_metrics(
                {
                    "train_accuracy": train_acc,
                    "test_accuracy": test_acc,
                    "test_f1_score": test_f1,
                },
                iteration=0,
            )

        # Make final predictions (if not already done for non-iterative models)
        if not (n_iterations and supports_warm_start):
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
        else:
            # For iterative models, predictions already done in the loop
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

        # Calculate probabilities for ROC-AUC (if available)
        try:
            y_test_proba = model.predict_proba(X_test)[:, 1]
            test_roc_auc = roc_auc_score(y_test, y_test_proba)
        except (AttributeError, IndexError):
            y_test_proba = None
            test_roc_auc = None

        # Calculate final metrics (including precision, recall, ROC-AUC)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)

        # Log final comprehensive metrics to ClearML (at final iteration)
        final_iteration = n_iterations if (n_iterations and supports_warm_start) else 0
        metrics = {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1_score": test_f1,
        }
        if test_roc_auc is not None:
            metrics["test_roc_auc"] = test_roc_auc

        logger.info(f"Final metrics: {metrics}")
        exp.log_metrics(metrics, iteration=final_iteration)

        # Log confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        cm_df = pd.DataFrame(
            cm,
            index=["No Churn", "Churn"],
            columns=["No Churn", "Churn"],
        )
        cm_path = MODELS_DIR / f"confusion_matrix_{model_type}_{exp.task.id}.csv"
        cm_df.to_csv(cm_path, index=True)
        exp.log_artifact("confusion_matrix", cm_path)

        # Also log confusion matrix as image and table for better visualization in ClearML UI
        exp.log_confusion_matrix(
            cm,
            class_names=["No Churn", "Churn"],
            title=f"Confusion Matrix - {exp.task_name}",
        )

        # Log feature importance (if available)
        if hasattr(model, "feature_importances_"):
            feature_importance = pd.DataFrame(
                {
                    "feature": X.columns,
                    "importance": model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)
            importance_path = MODELS_DIR / f"feature_importance_{model_type}_{exp.task.id}.csv"
            feature_importance.to_csv(importance_path, index=False)
            exp.log_artifact("feature_importance", importance_path)

        # Save and upload model
        model_path = MODELS_DIR / f"model_{model_type}_{exp.task.id}.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Saving model to {model_path}")
        exp.log_model("model", model_path)

        # Also upload model as output model for model registry
        try:
            exp.task.upload_artifact(name="output_model", artifact_object=str(model_path))
            logger.info("Model uploaded to ClearML successfully")
        except Exception as e:
            logger.warning(f"Could not upload model to ClearML: {e}")

        task_id = exp.task.id
        logger.info(f"Experiment completed. Task ID: {task_id}")
        logger.info(f"Test F1-Score: {test_f1:.4f}, Test Accuracy: {test_accuracy:.4f}")

        return {
            "task_id": task_id,
            "test_accuracy": test_accuracy,
            "test_f1_score": test_f1,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_roc_auc": test_roc_auc,
        }


def run_experiments(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    project_name: str = "Churn Prediction Experiments",
):
    """Run multiple experiments with different models and hyperparameters."""
    experiments = []

    # Random Forest experiments (5 experiments)
    experiments.extend(
        [
            {
                "model_type": "random_forest",
                "model_params": {"n_estimators": 50, "max_depth": 5, "class_weight": "balanced"},
                "run_name": "rf_n50_d5",
            },
            {
                "model_type": "random_forest",
                "model_params": {"n_estimators": 100, "max_depth": 8, "class_weight": "balanced"},
                "run_name": "rf_n100_d8",
            },
            {
                "model_type": "random_forest",
                "model_params": {"n_estimators": 200, "max_depth": 10, "class_weight": "balanced"},
                "run_name": "rf_n200_d10",
            },
            {
                "model_type": "random_forest",
                "model_params": {
                    "n_estimators": 100,
                    "max_depth": None,
                    "class_weight": "balanced",
                },
                "run_name": "rf_n100_unlimited",
            },
            {
                "model_type": "random_forest",
                "model_params": {"n_estimators": 150, "max_depth": 12, "class_weight": "balanced"},
                "run_name": "rf_n150_d12",
            },
        ]
    )

    # Gradient Boosting experiments (3 experiments)
    experiments.extend(
        [
            {
                "model_type": "gradient_boosting",
                "model_params": {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1},
                "run_name": "gb_n50_d3_lr01",
            },
            {
                "model_type": "gradient_boosting",
                "model_params": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
                "run_name": "gb_n100_d5_lr01",
            },
            {
                "model_type": "gradient_boosting",
                "model_params": {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.05},
                "run_name": "gb_n150_d5_lr005",
            },
        ]
    )

    # Logistic Regression experiments (3 experiments)
    experiments.extend(
        [
            {
                "model_type": "logistic_regression",
                "model_params": {"C": 0.1, "max_iter": 1000, "class_weight": "balanced"},
                "run_name": "lr_c01",
            },
            {
                "model_type": "logistic_regression",
                "model_params": {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"},
                "run_name": "lr_c1",
            },
            {
                "model_type": "logistic_regression",
                "model_params": {"C": 10.0, "max_iter": 1000, "class_weight": "balanced"},
                "run_name": "lr_c10",
            },
        ]
    )

    # SVM experiments (2 experiments)
    # Note: max_iter is limited to prevent timeout on large datasets
    experiments.extend(
        [
            {
                "model_type": "svm",
                "model_params": {
                    "C": 1.0,
                    "kernel": "rbf",
                    "class_weight": "balanced",
                    "max_iter": 500,
                },
                "run_name": "svm_rbf_c1",
            },
            {
                "model_type": "svm",
                "model_params": {
                    "C": 0.1,
                    "kernel": "linear",
                    "class_weight": "balanced",
                    "max_iter": 500,
                },
                "run_name": "svm_linear_c01",
            },
        ]
    )

    # KNN experiments (2 experiments)
    experiments.extend(
        [
            {
                "model_type": "knn",
                "model_params": {"n_neighbors": 5},
                "run_name": "knn_k5",
            },
            {
                "model_type": "knn",
                "model_params": {"n_neighbors": 10},
                "run_name": "knn_k10",
            },
        ]
    )

    # Decision Tree experiments (2 experiments)
    experiments.extend(
        [
            {
                "model_type": "decision_tree",
                "model_params": {"max_depth": 10, "class_weight": "balanced"},
                "run_name": "dt_d10",
            },
            {
                "model_type": "decision_tree",
                "model_params": {"max_depth": 15, "class_weight": "balanced"},
                "run_name": "dt_d15",
            },
        ]
    )

    # AdaBoost experiments (2 experiments)
    experiments.extend(
        [
            {
                "model_type": "adaboost",
                "model_params": {"n_estimators": 50, "learning_rate": 1.0},
                "run_name": "adaboost_n50",
            },
            {
                "model_type": "adaboost",
                "model_params": {"n_estimators": 100, "learning_rate": 0.5},
                "run_name": "adaboost_n100",
            },
        ]
    )

    # Improved models for better quality (replacing last 2 experiments)
    # Experiment 19: Enhanced Random Forest with optimized parameters
    experiments.append(
        {
            "model_type": "random_forest",
            "model_params": {
                "n_estimators": 300,
                "max_depth": 15,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "class_weight": "balanced",
                "n_jobs": -1,  # Use all CPU cores
            },
            "run_name": "rf_enhanced_n300_d15",
        }
    )

    # Experiment 20: Enhanced Gradient Boosting with optimized parameters
    experiments.append(
        {
            "model_type": "gradient_boosting",
            "model_params": {
                "n_estimators": 200,
                "max_depth": 7,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
            },
            "run_name": "gb_enhanced_n200_d7_lr005",
        }
    )

    logger.info(f"Starting {len(experiments)} experiments...")

    results = []
    for i, exp in enumerate(experiments, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment {i}/{len(experiments)}: {exp['run_name']}")
        logger.info(f"{'='*60}")

        try:
            result = train_and_log_experiment(
                model_type=cast(str, exp["model_type"]),
                model_params=cast(Dict[str, Any], exp["model_params"]),
                features_path=features_path,
                labels_path=labels_path,
                project_name=project_name,
                task_name=cast(str, exp["run_name"]),
            )
            result["run_name"] = exp["run_name"]
            result["model_type"] = exp["model_type"]
            results.append(result)
        except Exception as e:
            logger.error(f"Experiment {exp['run_name']} failed: {e}")
            continue

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENTS SUMMARY")
    logger.info("=" * 60)
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        logger.info(f"\nTotal experiments completed: {len(results_df)}")
        logger.info(f"Best F1-Score: {results_df['test_f1_score'].max():.4f}")
        logger.info(f"Best Accuracy: {results_df['test_accuracy'].max():.4f}")
        logger.info("\nTop 5 by F1-Score:")
        top5 = results_df.nlargest(5, "test_f1_score")[
            ["run_name", "model_type", "test_f1_score", "test_accuracy"]
        ]
        logger.info(f"\n{top5.to_string(index=False)}")

        # Compare experiments using ClearML utilities
        logger.info("\n" + "=" * 60)
        logger.info("CLEARML EXPERIMENT COMPARISON")
        logger.info("=" * 60)
        try:
            top_experiments = compare_experiments(
                project_name=project_name,
                metric_name="test_f1_score",
                top_n=5,
            )
            logger.info("\nTop 5 experiments from ClearML:")
            for exp in top_experiments:
                logger.info(
                    f"  {exp['task_name']}: F1={exp.get('test_f1_score', 'N/A'):.4f} "
                    f"(ID: {exp['task_id']})"
                )
        except Exception as e:
            logger.warning(f"Could not compare experiments in ClearML: {e}")

    return results


if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def main(
        features_path: Path = PROCESSED_DATA_DIR / "features.csv",
        labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
        project_name: str = "Churn Prediction Experiments",
    ):
        """Run multiple ML experiments with different models and hyperparameters."""
        run_experiments(features_path, labels_path, project_name)

    app()
