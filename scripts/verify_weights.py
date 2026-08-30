#!/usr/bin/env python3
"""Verify the released checkpoint identities and model-size contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "weights" / "manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = args.weights.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    expected = {row["name"]: row for row in manifest["files"]}
    observed_checkpoints = {
        path.name for path in root.iterdir() if path.is_file() and path.suffix in {".pt", ".pth"}
    }
    missing = sorted(set(expected) - observed_checkpoints)
    extra = sorted(observed_checkpoints - set(expected))
    if missing or extra:
        raise RuntimeError(f"Checkpoint set mismatch: missing={missing}, extra={extra}")

    total = 0
    for name, record in expected.items():
        path = root / name
        size = path.stat().st_size
        digest = sha256(path)
        if size != int(record["bytes"]):
            raise RuntimeError(f"Size mismatch for {name}: {size} != {record['bytes']}")
        if digest != record["sha256"]:
            raise RuntimeError(f"SHA256 mismatch for {name}: {digest}")
        total += size
        print(f"OK  {name}  {size} bytes")

    if total != int(manifest["total_model_bytes"]):
        raise RuntimeError(f"Total-size drift: {total}")
    if total >= int(manifest["maximum_model_bytes"]):
        raise RuntimeError(f"Model-size limit exceeded: {total}")
    print(f"Weight contract verified: {total} bytes")


if __name__ == "__main__":
    main()
