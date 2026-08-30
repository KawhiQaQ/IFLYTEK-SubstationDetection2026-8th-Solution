#!/usr/bin/env python3
"""Register custom DEIMv2 modules, then delegate to the upstream trainer."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
DEIM_ROOT = SOURCE_ROOT / "external" / "DEIMv2"


def main() -> None:
    sys.path.insert(0, str(SOURCE_ROOT))
    sys.path.insert(0, str(DEIM_ROOT))
    # Importing these modules registers their YAML-visible classes. The base
    # coordinate config does not require them, so one wrapper serves both runs.
    from workspace.models import deimv2_domain_generalized_boundary  # noqa: F401
    from workspace.models import deimv2_query_conditioned_boundary  # noqa: F401

    runpy.run_path(str(DEIM_ROOT / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
