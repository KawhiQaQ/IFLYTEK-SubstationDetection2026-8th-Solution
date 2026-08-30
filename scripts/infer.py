#!/usr/bin/env python3
"""Stable public entry point for the released final ensemble."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"


def main() -> None:
    sys.path.insert(0, str(SOURCE_ROOT))
    from infer_final_ensemble import main as inference_main

    inference_main()


if __name__ == "__main__":
    main()
