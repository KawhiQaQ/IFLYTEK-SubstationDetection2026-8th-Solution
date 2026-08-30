from __future__ import annotations

import ast
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryContractTest(unittest.TestCase):
    def test_weight_manifest_is_self_consistent(self) -> None:
        payload = json.loads((ROOT / "weights" / "manifest.json").read_text(encoding="utf-8"))
        rows = payload["files"]
        self.assertEqual(len(rows), 6)
        self.assertEqual(len({row["name"] for row in rows}), 6)
        self.assertEqual(sum(int(row["bytes"]) for row in rows), 568_789_683)
        self.assertEqual(payload["total_model_bytes"], 568_789_683)
        self.assertLess(payload["total_model_bytes"], payload["maximum_model_bytes"])
        for row in rows:
            self.assertEqual(len(row["sha256"]), 64)

    def test_inference_sources_parse(self) -> None:
        paths = [
            ROOT / "src" / "infer_parent_ensemble.py",
            ROOT / "src" / "infer_protected_ensemble.py",
            ROOT / "src" / "infer_final_ensemble.py",
            ROOT / "scripts" / "infer.py",
            ROOT / "scripts" / "verify_weights.py",
        ]
        for path in paths:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_required_inference_assets_exist(self) -> None:
        required = [
            ROOT / "src" / "rfdetr" / "__init__.py",
            ROOT / "src" / "external" / "DEIMv2" / "engine" / "core" / "yaml_config.py",
            ROOT / "src" / "workspace" / "models" / "rfdetr_query_consensus.py",
            ROOT / "src" / "workspace" / "models" / "boundary_refiner.py",
            ROOT
            / "src"
            / "workspace"
            / "configs"
            / "domain_generalized_specialist_full.yml",
        ]
        for path in required:
            self.assertTrue(path.is_file(), path)

    def test_public_sources_do_not_expose_internal_experiment_ids(self) -> None:
        pattern = re.compile(r"(?<![A-Za-z])[EeRr]\d{3}(?!\d)")
        roots = [ROOT / "README.md", ROOT / "README_CN.md", ROOT / "docs", ROOT / "scripts", ROOT / "training", ROOT / "weights", ROOT / "src" / "workspace"]
        text_suffixes = {".cff", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
        offenders: list[str] = []
        for root in roots:
            paths = [root] if root.is_file() else root.rglob("*")
            for path in paths:
                if not path.is_file() or path.suffix.lower() not in text_suffixes:
                    continue
                for number, line in enumerate(
                    path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
                ):
                    if "# noqa:" in line:
                        continue
                    if pattern.search(line):
                        offenders.append(f"{path.relative_to(ROOT)}:{number}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
