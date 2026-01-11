"""Create metrics directories for DVC pipeline."""

from pathlib import Path

# Create metrics directory
metrics_dir = Path("metrics")
metrics_dir.mkdir(exist_ok=True)

# Create initial metrics files
(metrics_dir / "data_download.json").write_text('{"status": "pending"}')
(metrics_dir / "features.json").write_text('{"status": "pending"}')

print(f"Created metrics directory: {metrics_dir}")
