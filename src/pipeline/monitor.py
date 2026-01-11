"""Pipeline monitoring and notification system."""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger


class PipelineMonitor:
    """Monitor pipeline execution and send notifications."""

    def __init__(self, log_file: Path = Path("pipeline_monitor.log")):
        """Initialize pipeline monitor.

        Args:
            log_file: Path to log file for monitoring events
        """
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.events: List[Dict[str, Any]] = []

    def log_stage_start(self, stage_name: str, **kwargs):
        """Log the start of a pipeline stage.

        Args:
            stage_name: Name of the stage
            **kwargs: Additional metadata
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event": "stage_start",
            "stage": stage_name,
            **kwargs,
        }
        self._log_event(event)
        logger.info(f"Pipeline stage started: {stage_name}")

    def log_stage_complete(self, stage_name: str, duration: float, **kwargs):
        """Log the completion of a pipeline stage.

        Args:
            stage_name: Name of the stage
            duration: Duration in seconds
            **kwargs: Additional metadata
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event": "stage_complete",
            "stage": stage_name,
            "duration_seconds": duration,
            **kwargs,
        }
        self._log_event(event)
        logger.info(f"Pipeline stage completed: {stage_name} (duration: {duration:.2f}s)")

    def log_stage_failed(self, stage_name: str, error: str, **kwargs):
        """Log the failure of a pipeline stage.

        Args:
            stage_name: Name of the stage
            error: Error message
            **kwargs: Additional metadata
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event": "stage_failed",
            "stage": stage_name,
            "error": str(error),
            **kwargs,
        }
        self._log_event(event)
        logger.error(f"Pipeline stage failed: {stage_name} - {error}")

    def log_metrics(self, stage_name: str, metrics: Dict[str, Any]):
        """Log metrics for a pipeline stage.

        Args:
            stage_name: Name of the stage
            metrics: Dictionary of metrics
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event": "metrics",
            "stage": stage_name,
            "metrics": metrics,
        }
        self._log_event(event)
        logger.info(f"Metrics logged for {stage_name}: {metrics}")

    def _log_event(self, event: Dict[str, Any]):
        """Log an event to file and memory.

        Args:
            event: Event dictionary
        """
        self.events.append(event)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(event) + "\n")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline execution.

        Returns:
            Dictionary with pipeline summary
        """
        stages_started = [e for e in self.events if e["event"] == "stage_start"]
        stages_completed = [e for e in self.events if e["event"] == "stage_complete"]
        stages_failed = [e for e in self.events if e["event"] == "stage_failed"]

        total_duration = sum(e.get("duration_seconds", 0) for e in stages_completed)

        return {
            "total_stages_started": len(stages_started),
            "total_stages_completed": len(stages_completed),
            "total_stages_failed": len(stages_failed),
            "total_duration_seconds": total_duration,
            "success_rate": len(stages_completed) / len(stages_started) if stages_started else 0,
            "events": self.events,
        }

    def save_summary(self, output_path: Path):
        """Save summary to file.

        Args:
            output_path: Path to save summary
        """
        summary = self.get_summary()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Pipeline summary saved to {output_path}")


def send_notification(message: str, level: str = "info"):
    """Send notification (placeholder for future integration with email/Slack/etc.).

    Args:
        message: Notification message
        level: Notification level (info, warning, error)
    """
    logger.log(level.upper(), f"NOTIFICATION: {message}")
    # Future: integrate with email, Slack, etc.


if __name__ == "__main__":
    # Example usage
    monitor = PipelineMonitor()
    monitor.log_stage_start("download_data")
    monitor.log_stage_complete("download_data", 5.2)
    monitor.log_metrics("download_data", {"rows": 1000, "size_mb": 2.5})

    summary = monitor.get_summary()
    print(json.dumps(summary, indent=2))
