#!/usr/bin/env python3
"""Initialize DVC if not already initialized."""

from pathlib import Path
import subprocess
import sys

dvc_dir = Path(".dvc")
config_file = dvc_dir / "config"

# Initialize DVC if not exists
if not dvc_dir.exists():
    print("Initializing DVC...")
    result = subprocess.run(["dvc", "init", "--no-scm"], capture_output=True)
    if result.returncode != 0:
        print(f"Error initializing DVC: {result.stderr.decode()}", file=sys.stderr)
        sys.exit(1)
    print("DVC initialized")
else:
    print("DVC already initialized")

# Add remote if not exists
if not config_file.exists() or "local" not in config_file.read_text():
    print("Configuring DVC remote...")
    result = subprocess.run(
        ["dvc", "remote", "add", "-d", "local", "./dvc_storage"],
        capture_output=True,
    )
    if result.returncode != 0:
        # Remote might already exist, check error message
        if "already exists" not in result.stderr.decode():
            print(f"Error configuring remote: {result.stderr.decode()}", file=sys.stderr)
            sys.exit(1)
    print("Remote configured")
else:
    print("Remote already configured")
