#!/usr/bin/env python3
"""
Download the public ONNX model bundle used by this reference implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download diarizen-onnx reference model files")
    parser.add_argument(
        "--repo-id",
        default="christopherthompson81/diarizen_onnx",
        help="Hugging Face repository containing the ONNX bundle",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./models"),
        help="Local output directory",
    )
    args = parser.parse_args()

    local_dir = args.output_dir.resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded model bundle to {local_dir}")


if __name__ == "__main__":
    main()
