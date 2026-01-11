"""ClearML Pipeline for churn prediction training and deployment."""

from pathlib import Path

from clearml import PipelineController, Task
from loguru import logger

# Import local config setup to ensure environment variables are set
from src.churn_prediction import clearml_local_config  # noqa: F401
from src.churn_prediction.config import ChurnPredictionConfig
from src.config import PROJ_ROOT


def _create_template_task(
    config: ChurnPredictionConfig,
    task_name: str,
    task_type: Task.TaskTypes,
    script_path: Path,
    params: dict,
):
    """Helper to create or find a template task."""
    # Try to find existing task first
    existing_tasks = Task.query_tasks(project_name=config.clearml_project, task_name=task_name)
    if existing_tasks:
        logger.info(f"Using existing task: {task_name} (ID: {existing_tasks[0].id})")
        return existing_tasks[0].id

    # Create new task
    task = Task.create(
        project_name=config.clearml_project,
        task_name=task_name,
        task_type=task_type,
    )

    for key, value in params.items():
        task.set_parameter(key, value)

    task.set_script(str(script_path))
    task.close()

    logger.info(f"Created template task: {task_name} (ID: {task.id})")
    return task.id


def create_template_tasks(config: ChurnPredictionConfig):
    """Create template tasks for pipeline steps.

    Args:
        config: Configuration object

    Returns:
        dict: Mapping of task names to task IDs
    """
    logger.info("Creating template tasks for pipeline steps...")

    template_tasks = {}

    # Task 1: Download dataset
    template_tasks["dataset_download.py"] = _create_template_task(
        config,
        "dataset_download.py",
        Task.TaskTypes.data_processing,
        PROJ_ROOT / "src" / "dataset.py",
        {"General/output_path": str(config.raw_data_dir / "customer_churn.csv")},
    )

    # Task 2: Create features
    template_tasks["features.py"] = _create_template_task(
        config,
        "features.py",
        Task.TaskTypes.data_processing,
        PROJ_ROOT / "src" / "features.py",
        {
            "General/input_path": str(config.raw_data_dir / "customer_churn.csv"),
            "General/features_path": str(config.processed_data_dir / "features.csv"),
            "General/labels_path": str(config.processed_data_dir / "labels.csv"),
        },
    )

    # Task 3: Train model
    template_tasks["train.py"] = _create_template_task(
        config,
        "train.py",
        Task.TaskTypes.training,
        PROJ_ROOT / "src" / "modeling" / "train.py",
        {
            "General/features_path": str(config.processed_data_dir / "features.csv"),
            "General/labels_path": str(config.processed_data_dir / "labels.csv"),
            "General/model_path": str(config.models_dir / "churn_model.pkl"),
            "General/n_estimators": 100,
            "General/max_depth": 8,
        },
    )

    # Task 4: Register model
    template_tasks["model_registry.py"] = _create_template_task(
        config,
        "model_registry.py",
        Task.TaskTypes.inference,
        PROJ_ROOT / "src" / "churn_prediction" / "model_registry.py",
        {
            "General/model_path": str(config.models_dir / "churn_model.pkl"),
            "General/model_name": "churn_prediction_model",
        },
    )

    logger.success(f"All {len(template_tasks)} template tasks ready")
    return template_tasks


def create_training_pipeline(config: ChurnPredictionConfig, template_tasks: dict[str, str]):
    """Create a ClearML pipeline for training churn prediction model.

    Args:
        config: Configuration object
    """
    logger.info("Creating ClearML training pipeline...")

    # Create pipeline controller
    # Note: Git warning is harmless - ClearML will detect git repo from project root
    pipe = PipelineController(
        name="Churn Prediction Training Pipeline",
        project=config.clearml_project,
        version="1.0.0",
    )

    # Step 1: Download dataset
    pipe.add_step(
        name="download_dataset",
        base_task_id=template_tasks["dataset_download.py"],
        parameter_override={
            "General/output_path": str(config.raw_data_dir / "customer_churn.csv"),
        },
    )

    # Step 2: Create features
    pipe.add_step(
        name="create_features",
        parents=["download_dataset"],
        base_task_id=template_tasks["features.py"],
        parameter_override={
            "General/input_path": str(config.raw_data_dir / "customer_churn.csv"),
            "General/features_path": str(config.processed_data_dir / "features.csv"),
            "General/labels_path": str(config.processed_data_dir / "labels.csv"),
        },
    )

    # Step 3: Train model
    pipe.add_step(
        name="train_model",
        parents=["create_features"],
        base_task_id=template_tasks["train.py"],
        parameter_override={
            "General/features_path": str(config.processed_data_dir / "features.csv"),
            "General/labels_path": str(config.processed_data_dir / "labels.csv"),
            "General/model_path": str(config.models_dir / "churn_model.pkl"),
            "General/n_estimators": 100,
            "General/max_depth": 8,
        },
    )

    # Step 4: Register model
    pipe.add_step(
        name="register_model",
        parents=["train_model"],
        base_task_id=template_tasks["model_registry.py"],
        parameter_override={
            "General/model_path": str(config.models_dir / "churn_model.pkl"),
            "General/model_name": "churn_prediction_model",
        },
    )

    logger.success("Pipeline created successfully")
    logger.info("Pipeline steps:")
    logger.info("  1. download_dataset")
    logger.info("  2. create_features (depends on: download_dataset)")
    logger.info("  3. train_model (depends on: create_features)")
    logger.info("  4. register_model (depends on: train_model)")

    return pipe


def create_deployment_pipeline(config: ChurnPredictionConfig, template_tasks: dict[str, str]):
    """Create a ClearML pipeline for deploying churn prediction model.

    Args:
        config: Configuration object
        template_tasks: Dictionary mapping task names to task IDs
    """
    logger.info("Creating ClearML deployment pipeline...")

    # Create pipeline controller
    # Note: Git warning is harmless - ClearML will detect git repo from project root
    pipe = PipelineController(
        name="Churn Prediction Deployment Pipeline",
        project=config.clearml_project,
        version="1.0.0",
    )

    # Step 1: Load model from registry
    # Note: For now, we'll skip deployment pipeline steps as they require additional scripts
    # that don't exist yet (model_load.py, deploy.py)
    logger.warning("Deployment pipeline steps require model_load.py and deploy.py scripts")
    logger.warning("Skipping deployment pipeline creation for now")

    return pipe


def main():
    """Main function to create and automatically start pipelines."""
    config = ChurnPredictionConfig()

    # Create template tasks first
    logger.info("Step 1: Creating template tasks...")
    template_tasks = create_template_tasks(config)

    logger.info("\nStep 2: Creating training pipeline...")
    training_pipeline = create_training_pipeline(config, template_tasks)
    logger.info(f"Training pipeline ID: {training_pipeline.id}")

    logger.info("\nStep 3: Creating deployment pipeline...")
    deployment_pipeline = create_deployment_pipeline(config, template_tasks)
    logger.info(f"Deployment pipeline ID: {deployment_pipeline.id}")

    logger.success("\n✓ Pipelines created successfully!")

    # Automatically queue the training pipeline
    logger.info("\nStep 4: Automatically queuing training pipeline...")
    try:
        training_pipeline.queue()
        logger.success("✓ Training pipeline queued successfully!")
        logger.info("Pipeline will be executed by ClearML Agent")
        logger.info(
            f"View pipeline status: http://localhost:8080/pipelines/{config.clearml_project}/experiments/{training_pipeline.id}"
        )
    except Exception as e:
        logger.warning(f"Failed to queue training pipeline: {e}")
        logger.info("You can manually queue it later using:")
        logger.info("  - ClearML UI: http://localhost:8080")
        logger.info("  - Or programmatically: training_pipeline.queue()")

    logger.info("\nNext steps:")
    logger.info("  1. View pipelines in ClearML UI:")
    logger.info(
        f"     Training: http://localhost:8080/pipelines/{config.clearml_project}/experiments/{training_pipeline.id}"
    )
    logger.info(
        f"     Deployment: http://localhost:8080/pipelines/{config.clearml_project}/experiments/{deployment_pipeline.id}"
    )
    logger.info("  2. Monitor pipeline execution in ClearML UI")
    logger.info("  3. To manually run pipelines:")
    logger.info("     - Queue: training_pipeline.queue()")
    logger.info("     - Run locally: training_pipeline.start()")


if __name__ == "__main__":
    main()
