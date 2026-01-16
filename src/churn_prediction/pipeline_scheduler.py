"""Automatic pipeline execution scheduler for ClearML pipelines.

This module provides scheduling functionality that complements ClearML's
built-in scheduling (available in Enterprise version).

For Enterprise users: Consider using ClearML's Task Scheduler application
through the UI, which provides more advanced scheduling features.

This module is useful for:
- Open-source ClearML users without Enterprise scheduling
- Custom scheduling requirements
- Integration with external scheduling systems
"""

import os
import time
from datetime import datetime, timedelta
from typing import Optional

from clearml import Task
from loguru import logger

from src.churn_prediction.config import ChurnPredictionConfig
from src.churn_prediction.pipeline import create_pipeline


class PipelineScheduler:
    """Schedule automatic pipeline execution."""

    def __init__(
        self,
        schedule_type: str = "interval",
        interval_seconds: Optional[int] = None,
        schedule_time: Optional[str] = None,
        max_runs: Optional[int] = None,
    ):
        """Initialize pipeline scheduler.

        Args:
            schedule_type: Type of schedule - "interval" or "daily"
            interval_seconds: Interval in seconds for interval scheduling
            schedule_time: Time of day for daily scheduling (HH:MM format)
            max_runs: Maximum number of pipeline runs (None for unlimited)
        """
        self.schedule_type = schedule_type
        self.interval_seconds = interval_seconds or 3600  # Default: 1 hour
        self.schedule_time = schedule_time or "00:00"  # Default: midnight
        self.max_runs = max_runs
        self.run_count = 0
        self.config = ChurnPredictionConfig()

    def schedule_interval(self):
        """Schedule pipeline to run at regular intervals."""
        logger.info(f"Starting interval scheduler (interval: {self.interval_seconds}s)")
        
        while True:
            if self.max_runs and self.run_count >= self.max_runs:
                logger.info(f"Reached maximum runs ({self.max_runs}). Stopping scheduler.")
                break

            try:
                logger.info(f"Starting pipeline run #{self.run_count + 1}")
                pipeline = create_pipeline()
                self.run_count += 1
                logger.info(f"Pipeline queued successfully. Run #{self.run_count}")
                logger.info(f"Next run in {self.interval_seconds} seconds")
                
                time.sleep(self.interval_seconds)
            except KeyboardInterrupt:
                logger.info("Scheduler interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in scheduled pipeline execution: {e}")
                logger.info(f"Retrying in {self.interval_seconds} seconds...")
                time.sleep(self.interval_seconds)

    def schedule_daily(self):
        """Schedule pipeline to run daily at specified time."""
        logger.info(f"Starting daily scheduler (time: {self.schedule_time})")
        
        # Parse schedule time
        try:
            hour, minute = map(int, self.schedule_time.split(":"))
        except ValueError:
            logger.error(f"Invalid schedule_time format: {self.schedule_time}. Use HH:MM")
            return

        while True:
            if self.max_runs and self.run_count >= self.max_runs:
                logger.info(f"Reached maximum runs ({self.max_runs}). Stopping scheduler.")
                break

            # Calculate next run time
            now = datetime.now()
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            # If time has passed today, schedule for tomorrow
            if next_run <= now:
                next_run += timedelta(days=1)

            wait_seconds = (next_run - now).total_seconds()
            logger.info(f"Next pipeline run scheduled for: {next_run.isoformat()}")
            logger.info(f"Waiting {wait_seconds:.0f} seconds ({wait_seconds/3600:.1f} hours)")

            # Wait until scheduled time
            time.sleep(wait_seconds)

            try:
                logger.info(f"Starting scheduled pipeline run #{self.run_count + 1}")
                pipeline = create_pipeline()
                self.run_count += 1
                logger.info(f"Pipeline queued successfully. Run #{self.run_count}")
            except KeyboardInterrupt:
                logger.info("Scheduler interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in scheduled pipeline execution: {e}")

    def start(self):
        """Start the scheduler."""
        logger.info("=" * 60)
        logger.info("Pipeline Scheduler Starting")
        logger.info("=" * 60)
        logger.info(f"Schedule type: {self.schedule_type}")
        logger.info(f"Max runs: {self.max_runs or 'unlimited'}")
        logger.info("=" * 60)

        if self.schedule_type == "interval":
            self.schedule_interval()
        elif self.schedule_type == "daily":
            self.schedule_daily()
        else:
            logger.error(f"Unknown schedule type: {self.schedule_type}")
            logger.error("Supported types: 'interval', 'daily'")

    @staticmethod
    def create_clearml_scheduled_task(
        schedule_expression: str,
        pipeline_script: str = "src/churn_prediction/pipeline.py",
    ):
        """Create a task in ClearML that can be scheduled externally.

        Note: This does NOT create a scheduled task in ClearML (which requires
        Enterprise features). Instead, it creates a task that can be scheduled
        using external tools (cron, systemd, Windows Task Scheduler, etc.).

        For Enterprise users: Use ClearML's Task Scheduler application in the UI
        for native scheduling support.

        Args:
            schedule_expression: Cron-like schedule expression (for documentation only)
            pipeline_script: Path to pipeline script

        Returns:
            ClearML Task object that can be executed by external schedulers
        """
        config = ChurnPredictionConfig()
        
        # Create a task in ClearML (not a scheduled task - that requires Enterprise)
        task = Task.init(
            project_name=config.clearml_project,
            task_name="Scheduled Pipeline Execution",
            task_type=Task.TaskTypes.controller,
            reuse_last_task_id=False,
        )

        # Set the script to run
        task.set_script(pipeline_script)

        logger.info(f"Created task for external scheduling (expression: {schedule_expression})")
        logger.warning("Note: This task is NOT automatically scheduled in ClearML")
        logger.info("To schedule: Use external scheduler (cron, systemd, Windows Task Scheduler)")
        logger.info("Or use ClearML Enterprise Task Scheduler application in UI")

        return task


def main():
    """Main function for pipeline scheduler CLI.
    
    Usage:
        # Basic usage
        python pipeline_scheduler.py interval 3600
        
        # With max runs
        python pipeline_scheduler.py interval 3600 --max-runs 5
        
        # Daily schedule
        python pipeline_scheduler.py daily 00:00
        
        # Using environment variables (Linux/Mac)
        PIPELINE_MAX_RUNS=5 python pipeline_scheduler.py interval 20
        
        # Using environment variables (Windows PowerShell)
        $env:PIPELINE_MAX_RUNS=5; python pipeline_scheduler.py interval 20
    """
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="ClearML Pipeline Scheduler")
    parser.add_argument("schedule_type", nargs="?", default="interval", 
                       choices=["interval", "daily"],
                       help="Schedule type: interval or daily")
    parser.add_argument("schedule_value", nargs="?", type=str, default="3600",
                       help="For interval: seconds. For daily: time in HH:MM format")
    parser.add_argument("--max-runs", type=int, default=None,
                       help="Maximum number of pipeline runs (0 or None for unlimited)")
    parser.add_argument("--interval-seconds", type=int, default=None,
                       help="Interval in seconds (overrides schedule_value for interval)")
    parser.add_argument("--schedule-time", type=str, default=None,
                       help="Time in HH:MM format (overrides schedule_value for daily)")
    
    args = parser.parse_args()
    
    # Get from environment if not provided via args
    max_runs = args.max_runs
    if max_runs is None:
        max_runs_env = os.getenv("PIPELINE_MAX_RUNS", "0")
        max_runs = int(max_runs_env) if max_runs_env else None
        if max_runs == 0:
            max_runs = None
    
    # Parse schedule value
    if args.schedule_type == "interval":
        interval_seconds = args.interval_seconds or int(args.schedule_value)
        schedule_time = "00:00"
    else:  # daily
        schedule_time = args.schedule_time or args.schedule_value
        interval_seconds = 3600  # default, not used for daily

    scheduler = PipelineScheduler(
        schedule_type=args.schedule_type,
        interval_seconds=interval_seconds,
        schedule_time=schedule_time,
        max_runs=max_runs,
    )

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")


if __name__ == "__main__":
    main()
