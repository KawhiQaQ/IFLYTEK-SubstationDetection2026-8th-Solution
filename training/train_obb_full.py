#!/usr/bin/env python3
"""Train the selected DOTA-pretrained OBB detector on all official labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    for path in (args.weights, args.data):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest_path = args.data.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("train_images") != 1473 or not manifest.get("train_only_fixed_epoch"):
        raise RuntimeError(f"Unexpected full-data manifest: {manifest}")
    if manifest.get("round1_test_images_read") or manifest.get("round2_test_images_read"):
        raise RuntimeError("Competition test data must not be read")

    args.report.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    if getattr(model, "task", None) != "obb":
        raise RuntimeError(f"Expected OBB checkpoint, got task={model.task}")
    parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
    preflight = {
        "component": "dota_obb_specialist_full",
        "initial_checkpoint": str(args.weights),
        "initial_checkpoint_sha256": sha256(args.weights),
        "parameters": parameter_count,
        "task": model.task,
        "train_images": 1473,
        "selected_completed_epochs_from_fold0": 27,
        "validation_used_for_selection": False,
        "round1_test_images_read": 0,
        "round2_test_images_read": 0,
    }
    (args.report / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    print(json.dumps(preflight), flush=True)

    run_name = "yolo26x_obb_dota_full"
    model.train(
        data=str(args.data),
        project=str(args.project),
        name=run_name,
        exist_ok=False,
        epochs=27,
        patience=0,
        imgsz=1024,
        batch=1,
        device=0,
        workers=8,
        cache="disk",
        pretrained=True,
        optimizer="AdamW",
        lr0=0.0001,
        lrf=0.10,
        momentum=0.90,
        weight_decay=0.0005,
        warmup_epochs=1.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.05,
        cos_lr=True,
        nbs=64,
        single_cls=True,
        deterministic=True,
        seed=3837,
        amp=True,
        compile=False,
        rect=False,
        multi_scale=0.0,
        close_mosaic=8,
        hsv_h=0.01,
        hsv_s=0.25,
        hsv_v=0.18,
        degrees=0.0,
        translate=0.08,
        scale=0.30,
        shear=0.0,
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,
        bgr=0.0,
        mosaic=0.5,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        val=False,
        iou=0.7,
        max_det=20,
        save=True,
        save_period=-1,
        plots=False,
        verbose=True,
    )
    run_dir = args.project / run_name
    final_checkpoint = run_dir / "weights" / "last.pt"
    if not final_checkpoint.is_file():
        raise FileNotFoundError(final_checkpoint)
    result = {
        **preflight,
        "final_checkpoint": str(final_checkpoint),
        "final_checkpoint_sha256": sha256(final_checkpoint),
        "completed_epochs": 27,
    }
    (args.report / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
