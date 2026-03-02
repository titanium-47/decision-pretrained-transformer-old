#!/usr/bin/env python3
"""
Recursively delete model checkpoints in context_results, keeping only model_epoch_18.pth.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def cleanup_checkpoints(root: Path, keep_name: str, dry_run: bool) -> tuple[int, int]:
    scanned = 0
    deleted = 0

    for checkpoint_path in root.rglob("model_epoch_*.pth"):
        if not checkpoint_path.is_file():
            continue

        scanned += 1
        if checkpoint_path.name == keep_name:
            continue

        if dry_run:
            print(f"[DRY RUN] Would delete: {checkpoint_path}")
        else:
            checkpoint_path.unlink()
            print(f"Deleted: {checkpoint_path}")
        deleted += 1

    return scanned, deleted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete model_epoch_*.pth recursively under context_results, "
            "keeping only model_epoch_18.pth."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("context_results"),
        help="Root directory to search (default: context_results).",
    )
    parser.add_argument(
        "--keep",
        default="model_epoch_18.pth",
        help="Checkpoint filename to keep (default: model_epoch_18.pth).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which files would be deleted without deleting them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root

    if not root.exists():
        raise SystemExit(f"Root directory does not exist: {root}")
    if not root.is_dir():
        raise SystemExit(f"Root path is not a directory: {root}")

    scanned, deleted = cleanup_checkpoints(root=root, keep_name=args.keep, dry_run=args.dry_run)
    print(f"Scanned checkpoints: {scanned}")
    print(f"Deleted checkpoints: {deleted}")
    print(f"Kept filename: {args.keep}")


if __name__ == "__main__":
    main()
