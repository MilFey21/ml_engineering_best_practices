#!/usr/bin/env python3
"""Check DVC file information and compare hashes (works without Git)."""

import hashlib
from pathlib import Path
import sys

import yaml


def calculate_md5(file_path: Path) -> str:
    """Calculate MD5 hash of a file."""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def get_file_size(file_path: Path) -> int:
    """Get file size in bytes."""
    return file_path.stat().st_size


def check_dvc_file(dvc_file_path: Path) -> None:
    """Check DVC file information and compare with actual file."""
    if not dvc_file_path.exists():
        print(f"Error: DVC file not found: {dvc_file_path}", file=sys.stderr)
        sys.exit(1)

    # Read DVC file
    with open(dvc_file_path, "r") as f:
        dvc_data = yaml.safe_load(f)

    if "outs" not in dvc_data or not dvc_data["outs"]:
        print(f"Error: Invalid DVC file format: {dvc_file_path}", file=sys.stderr)
        sys.exit(1)

    dvc_info = dvc_data["outs"][0]
    expected_hash = dvc_info.get("md5", "")
    expected_size = dvc_info.get("size", 0)
    file_path = dvc_file_path.parent / dvc_info.get("path", "")

    if not file_path.exists():
        print(f"File: {file_path}")
        print("Status: File not found")
        print(f"Expected hash (from .dvc): {expected_hash}")
        print(f"Expected size: {expected_size} bytes")
        return

    # Calculate current file hash and size
    current_hash = calculate_md5(file_path)
    current_size = get_file_size(file_path)

    # Compare
    hash_match = current_hash == expected_hash
    size_match = current_size == expected_size

    print(f"File: {file_path}")
    print(f"Expected hash (from .dvc): {expected_hash}")
    print(f"Current hash:              {current_hash}")
    print(f"Hash match: {'✓' if hash_match else '✗'}")
    print(f"Expected size: {expected_size} bytes")
    print(f"Current size:  {current_size} bytes")
    print(f"Size match: {'✓' if size_match else '✗'}")

    if hash_match and size_match:
        print("\nStatus: File matches DVC record (no changes)")
    else:
        print("\nStatus: File differs from DVC record")
        if not hash_match:
            print("  - Hash mismatch: file content has changed")
        if not size_match:
            print("  - Size mismatch: file size has changed")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/dvc_diff.py <dvc_file_path>", file=sys.stderr)
        print(
            "Example: python scripts/dvc_diff.py data/raw/customer_churn.csv.dvc", file=sys.stderr
        )
        print("Or use: pixi run dvc-diff data/raw/customer_churn.csv", file=sys.stderr)
        sys.exit(1)

    file_arg = sys.argv[1]

    # Handle both .dvc file path and data file path
    if file_arg.endswith(".dvc"):
        dvc_file_path = Path(file_arg)
    else:
        # If data file path provided, find corresponding .dvc file
        data_file_path = Path(file_arg)
        dvc_file_path = data_file_path.with_suffix(data_file_path.suffix + ".dvc")

    check_dvc_file(dvc_file_path)


if __name__ == "__main__":
    main()
