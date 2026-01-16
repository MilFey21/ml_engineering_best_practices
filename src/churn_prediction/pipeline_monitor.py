"""Pipeline monitoring and notification system for ClearML pipelines.

This module provides additional functionality on top of ClearML's built-in UI monitoring:
- Automated notifications (file-based and webhook)
- Integration with external systems
- Programmatic status polling for automation

Note: ClearML UI already provides pipeline monitoring. This module adds:
- Custom notification channels (webhooks, file logs)
- Integration with external alerting systems
- Automated monitoring scripts for CI/CD pipelines
"""

import os
import time
from datetime import datetime
from typing import Dict, List, Optional

from clearml import PipelineController, Task
from loguru import logger


class PipelineMonitor:
    """Monitor ClearML pipeline execution and send notifications."""

    def __init__(
        self,
        project_name: str = "Churn Prediction",
        pipeline_name: str = "Churn Prediction Pipeline",
        check_interval: int = 30,
        log_file: Optional[str] = None,
    ):
        """Initialize pipeline monitor.

        Args:
            project_name: ClearML project name
            pipeline_name: Pipeline name to monitor
            check_interval: Interval in seconds between status checks
            log_file: Optional file path for logging pipeline status
        """
        self.project_name = project_name
        self.pipeline_name = pipeline_name
        self.check_interval = check_interval
        self.log_file = log_file or "pipeline_monitor.log"

    def get_pipeline_status(self, pipeline_id: Optional[str] = None) -> Dict:
        """Get current status of a pipeline using ClearML API.

        Note: This uses ClearML's built-in API methods. The UI provides the same
        information, but this allows programmatic access for automation.

        Args:
            pipeline_id: Optional pipeline ID. If None, gets latest pipeline.

        Returns:
            Dictionary with pipeline status information
        """
        try:
            if pipeline_id:
                # Get specific pipeline using ClearML API
                pipeline = PipelineController.get_pipeline(pipeline_id)
            else:
                # Get latest pipeline by name using ClearML API
                tasks = Task.get_tasks(
                    project_name=self.project_name,
                    task_name=self.pipeline_name,
                    task_type=Task.TaskTypes.controller,
                    order_by=["-created"],
                    page_size=1,
                )
                if not tasks:
                    return {
                        "status": "not_found",
                        "message": f"Pipeline '{self.pipeline_name}' not found",
                    }

                task = tasks[0]
                pipeline = PipelineController.get_pipeline(task.id)

            # Use ClearML's built-in status methods
            status = pipeline.get_status()
            steps_status = pipeline.get_steps_status()

            return {
                "pipeline_id": pipeline.id,
                "status": status,
                "steps": steps_status,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error getting pipeline status: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def monitor_pipeline(
        self,
        pipeline_id: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Dict:
        """Monitor pipeline execution until completion or timeout.

        This provides automated monitoring with notifications, which complements
        ClearML UI's real-time monitoring. Useful for CI/CD integration and
        automated alerting.

        Args:
            pipeline_id: Optional pipeline ID to monitor
            timeout: Optional timeout in seconds. If None, monitors indefinitely.

        Returns:
            Final pipeline status
        """
        logger.info(f"Starting monitoring for pipeline: {self.pipeline_name}")
        start_time = time.time()

        while True:
            status_info = self.get_pipeline_status(pipeline_id)

            # Log status
            self._log_status(status_info)

            # Check if pipeline is complete
            if status_info.get("status") in ["completed", "failed", "stopped"]:
                logger.info(f"Pipeline finished with status: {status_info['status']}")
                self._send_notification(status_info)
                return status_info

            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                logger.warning(f"Monitoring timeout reached ({timeout}s)")
                status_info["status"] = "timeout"
                self._send_notification(status_info)
                return status_info

            # Wait before next check
            time.sleep(self.check_interval)

    def _log_status(self, status_info: Dict):
        """Log pipeline status to file and console."""
        status = status_info.get("status", "unknown")
        pipeline_id = status_info.get("pipeline_id", "N/A")
        timestamp = status_info.get("timestamp", datetime.now().isoformat())

        log_message = (
            f"[{timestamp}] Pipeline {pipeline_id}: {status}\n"
        )

        # Log to file
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_message)
                if "steps" in status_info:
                    for step_name, step_status in status_info["steps"].items():
                        f.write(f"  Step '{step_name}': {step_status}\n")
        except Exception as e:
            logger.warning(f"Failed to write to log file: {e}")

        # Log to console
        logger.info(f"Pipeline {pipeline_id} status: {status}")
        if "steps" in status_info:
            for step_name, step_status in status_info["steps"].items():
                logger.info(f"  Step '{step_name}': {step_status}")

    def _send_notification(self, status_info: Dict):
        """Send notification about pipeline status.

        Args:
            status_info: Pipeline status information
        """
        status = status_info.get("status", "unknown")
        pipeline_id = status_info.get("pipeline_id", "N/A")
        timestamp = status_info.get("timestamp", datetime.now().isoformat())

        # Create notification message
        message = f"""
Pipeline Status Notification
============================
Pipeline: {self.pipeline_name}
Pipeline ID: {pipeline_id}
Status: {status}
Time: {timestamp}
"""

        if "steps" in status_info:
            message += "\nSteps Status:\n"
            for step_name, step_status in status_info["steps"].items():
                message += f"  - {step_name}: {step_status}\n"

        # Log notification
        logger.info("=" * 60)
        logger.info("PIPELINE NOTIFICATION")
        logger.info("=" * 60)
        logger.info(message)
        logger.info("=" * 60)

        # Write to notification file
        notification_file = "pipeline_notifications.log"
        try:
            with open(notification_file, "a", encoding="utf-8") as f:
                f.write(message)
                f.write("\n" + "=" * 60 + "\n\n")
        except Exception as e:
            logger.warning(f"Failed to write notification: {e}")

        # Check for webhook URL in environment
        webhook_url = os.getenv("CLEARML_WEBHOOK_URL")
        if webhook_url:
            self._send_webhook_notification(webhook_url, status_info, message)

    def _send_webhook_notification(
        self, webhook_url: str, status_info: Dict, message: str
    ):
        """Send webhook notification (if configured).

        Args:
            webhook_url: Webhook URL
            status_info: Pipeline status information
            message: Notification message
        """
        try:
            import requests

            payload = {
                "pipeline_name": self.pipeline_name,
                "pipeline_id": status_info.get("pipeline_id"),
                "status": status_info.get("status"),
                "timestamp": status_info.get("timestamp"),
                "message": message,
                "steps": status_info.get("steps", {}),
            }

            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info(f"Webhook notification sent successfully to {webhook_url}")
        except ImportError:
            logger.warning("requests library not installed. Webhook notifications disabled.")
        except Exception as e:
            logger.warning(f"Failed to send webhook notification: {e}")

    def list_pipelines(self) -> List[Dict]:
        """List all pipelines in the project using ClearML API.

        Note: This provides programmatic access. The same information is
        available in ClearML UI under Pipelines section.

        Returns:
            List of pipeline information dictionaries
        """
        try:
            tasks = Task.get_tasks(
                project_name=self.project_name,
                task_type=Task.TaskTypes.controller,
                order_by=["-created"],
                page_size=50,
            )

            pipelines = []
            for task in tasks:
                try:
                    pipeline = PipelineController.get_pipeline(task.id)
                    status = pipeline.get_status()
                    pipelines.append({
                        "pipeline_id": task.id,
                        "name": task.name,
                        "status": status,
                        "created": task.created.isoformat() if task.created else None,
                    })
                except Exception as e:
                    logger.warning(f"Failed to get pipeline {task.id}: {e}")

            return pipelines
        except Exception as e:
            logger.error(f"Error listing pipelines: {e}")
            return []


def main():
    """Main function for pipeline monitoring CLI."""
    import sys

    if len(sys.argv) > 1:
        pipeline_id = sys.argv[1]
    else:
        pipeline_id = None

    monitor = PipelineMonitor()
    
    if pipeline_id:
        logger.info(f"Monitoring pipeline: {pipeline_id}")
        status = monitor.monitor_pipeline(pipeline_id=pipeline_id)
    else:
        logger.info("Listing all pipelines...")
        pipelines = monitor.list_pipelines()
        if pipelines:
            logger.info(f"Found {len(pipelines)} pipelines:")
            for p in pipelines:
                logger.info(f"  - {p['name']} ({p['pipeline_id']}): {p['status']}")
            
            # Monitor the latest pipeline
            if pipelines:
                latest = pipelines[0]
                logger.info(f"\nMonitoring latest pipeline: {latest['pipeline_id']}")
                status = monitor.monitor_pipeline(pipeline_id=latest["pipeline_id"])
        else:
            logger.warning("No pipelines found")


if __name__ == "__main__":
    main()
