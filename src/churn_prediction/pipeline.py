"""Simple ClearML pipeline for churn prediction workflow."""

import os
import time

from clearml import PipelineController, Task
from loguru import logger

from src.churn_prediction.config import ChurnPredictionConfig


def create_pipeline():
    """Create and queue a ClearML pipeline for churn prediction."""
    config = ChurnPredictionConfig()
    execution_queue = os.getenv("CLEARML_EXECUTION_QUEUE", "default")
    
    # Initialize ClearML
    Task.init(
        project_name=config.clearml_project,
        task_name="Pipeline Controller",
        task_type=Task.TaskTypes.controller,
        reuse_last_task_id=False,
    )
    
    # Create pipeline
    pipe = PipelineController(
        name="Churn Prediction Pipeline",
        project=config.clearml_project,
        version="1.0.0",
    )
    pipe.set_default_execution_queue(execution_queue)
    
    # Helper to create task factory
    def create_task_factory(script_path: str, task_name: str):
        def factory():
            return Task.init(
                project_name=config.clearml_project,
                task_name=task_name,
                script=script_path,
                reuse_last_task_id=False,
            )
        return factory
    
    # Add pipeline steps
    logger.info("Creating pipeline steps...")
    pipe.add_step(
        name="download_data",
        base_task_factory=create_task_factory("src/dataset.py", "Download Dataset"),
        parameter_override={"General/output_path": "data/raw/customer_churn.csv"},
    )
    logger.info("  ✓ Step 1: download_data")
    
    pipe.add_step(
        name="create_features",
        base_task_factory=create_task_factory("src/features.py", "Create Features"),
        parents=["download_data"],
        parameter_override={
            "General/input_path": "data/raw/customer_churn.csv",
            "General/features_path": "data/processed/features.csv",
            "General/labels_path": "data/processed/labels.csv",
        },
    )
    logger.info("  ✓ Step 2: create_features (depends on: download_data)")
    
    pipe.add_step(
        name="train_model",
        base_task_factory=create_task_factory("src/modeling/train_with_hydra.py", "Train Model"),
        parents=["create_features"],
        parameter_override={"General/model": "random_forest"},
    )
    logger.info("  ✓ Step 3: train_model (depends on: create_features)")
    
    # Start pipeline
    logger.info(f"\nStarting pipeline (queue: {execution_queue})...")
    pipe.start(queue=execution_queue)
    
    logger.info(f"Pipeline ID: {pipe.id}")
    
    # Get UI URL from environment or use default
    web_host = os.getenv("CLEARML_WEB_HOST", "http://localhost:8080")
    logger.info(f"View in UI: {web_host}")
    
    # Check initial status
    initial_status = pipe.get_status()
    if initial_status.lower() == "pending":
        logger.warning("\n⚠️  Pipeline is in PENDING status - waiting for agent to pick it up")
        logger.warning(f"   Queue: {execution_queue}")
        logger.warning("   To start execution, run in a separate terminal:")
        logger.warning(f"   clearml-agent daemon --queue {execution_queue} --create-queue")
        logger.warning("")
    
    # Monitor pipeline execution
    _monitor_pipeline(pipe, execution_queue)
    
    return pipe


def _monitor_pipeline(pipe: PipelineController, queue_name: str, check_interval: int = 5, max_pending_checks: int = 12):
    """Monitor pipeline execution and display progress in terminal."""
    logger.info("\nMonitoring pipeline execution...")
    logger.info("Press Ctrl+C to stop monitoring (pipeline will continue running)\n")
    
    pending_count = 0
    web_host = os.getenv("CLEARML_WEB_HOST", "http://localhost:8080")
    
    try:
        while True:
            status = pipe.get_status()
            steps = pipe.get_steps_status()
            
            # Clear previous output
            print("\033[2J\033[H", end="")
            
            logger.info("=" * 70)
            logger.info("Pipeline Execution Status")
            logger.info("=" * 70)
            logger.info(f"Pipeline ID: {pipe.id}")
            logger.info(f"Status: {status}")
            logger.info(f"Queue: {queue_name}")
            logger.info("")
            logger.info("Steps:")
            
            all_pending = True
            for step_name, step_status in steps.items():
                status_icon = {
                    "pending": "⏳",
                    "running": "🔄",
                    "completed": "✅",
                    "failed": "❌",
                    "stopped": "⏹️",
                }.get(step_status.lower(), "❓")
                logger.info(f"  {status_icon} {step_name}: {step_status}")
                if step_status.lower() != "pending":
                    all_pending = False
            
            logger.info("")
            
            # Show warning if stuck in pending
            if status.lower() == "pending" or all_pending:
                pending_count += 1
                if pending_count >= max_pending_checks:
                    logger.warning("=" * 70)
                    logger.warning("⚠️  Pipeline is still PENDING - no agent is executing it")
                    logger.warning("")
                    logger.warning("To start execution, run in a separate terminal:")
                    logger.warning(f"  clearml-agent daemon --queue {queue_name} --create-queue")
                    logger.warning("")
                    logger.warning("Or install agent first:")
                    logger.warning("  pip install clearml-agent")
                    logger.warning("")
                    logger.warning(f"Check queue status in UI: {web_host} → Workers & Queues")
                    logger.warning("=" * 70)
                    pending_count = 0  # Reset counter
            else:
                pending_count = 0
            
            logger.info("=" * 70)
            
            # Check if pipeline is done
            if status.lower() in ["completed", "failed", "stopped"]:
                if status.lower() == "completed":
                    logger.success("\n✅ Pipeline completed successfully!")
                else:
                    logger.warning(f"\n⚠️  Pipeline finished with status: {status}")
                break
            
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        logger.info("\n\nMonitoring stopped. Pipeline continues running in background.")
        logger.info(f"Check status in UI: {web_host}")


def main():
    """Main function to create and start pipeline."""
    try:
        pipe = create_pipeline()
        logger.success("\nPipeline created and started successfully!")
    except Exception as e:
        logger.error(f"Pipeline creation failed: {e}")
        raise


if __name__ == "__main__":
    main()
