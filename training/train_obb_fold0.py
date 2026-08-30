#!/usr/bin/env python3
"""Fine-tune the complete DOTA-pretrained YOLO26x-OBB and export strict HBB predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from faster_coco_eval import COCO, COCOeval_faster
from ultralytics import YOLO


IOU_THRESHOLDS = np.arange(0.50, 0.951, 0.05, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--val-list", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Export fixed validation predictions from --weights without training.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(annotations: Path, predictions: list[dict]) -> dict:
    coco_gt = COCO(str(annotations))
    evaluator = COCOeval_faster(
        coco_gt,
        coco_gt.loadRes(predictions),
        iouType="bbox",
        print_function=lambda *_args, **_kwargs: None,
        separate_eval=True,
    )
    evaluator.params.maxDets = [1, 10, 20]
    evaluator.evaluate()
    evaluator.accumulate()
    precision = np.asarray(evaluator.eval["precision"], dtype=np.float64)
    values: list[float] = []
    for index in range(len(IOU_THRESHOLDS)):
        valid = precision[index, :, :, 0, -1]
        valid = valid[valid > -1]
        values.append(float(valid.mean()))
    return {
        "map_50_95": float(np.mean(values)),
        "ap_95": values[-1],
        "ap_by_iou": {
            f"{threshold:.2f}": value
            for threshold, value in zip(IOU_THRESHOLDS, values, strict=True)
        },
    }


def export_predictions(model: YOLO, val_list: Path, annotations: Path) -> list[dict]:
    images = [Path(line.strip()).resolve() for line in val_list.read_text().splitlines() if line.strip()]
    annotation_payload = json.loads(annotations.read_text(encoding="utf-8"))
    id_by_stem = {Path(row["file_name"]).stem: int(row["id"]) for row in annotation_payload["images"]}
    if {image.stem for image in images} != set(id_by_stem):
        raise RuntimeError("Validation list does not exactly match fold0 annotations")
    predictions: list[dict] = []
    # Passing a Python list makes Ultralytics materialize every image as one
    # enormous in-memory batch. A text source keeps the same ordered paths but
    # uses the streaming file loader (batch size 1).
    results = model.predict(
        source=str(val_list.resolve()),
        imgsz=1024,
        conf=0.001,
        iou=0.7,
        max_det=20,
        device=0,
        half=True,
        augment=False,
        agnostic_nms=True,
        verbose=False,
        stream=True,
    )
    seen: set[str] = set()
    for result in results:
        stem = Path(result.path).stem
        if stem not in id_by_stem or stem in seen:
            raise RuntimeError(f"Unexpected or duplicate validation image: {stem}")
        seen.add(stem)
        if result.obb is None:
            continue
        corners = result.obb.xyxyxyxy.detach().float().cpu().numpy()
        scores = result.obb.conf.detach().float().cpu().numpy()
        order = np.argsort(-scores, kind="stable")[:20]
        for index in order:
            # The competition evaluates axis-aligned boxes. Convert an OBB to
            # its exact enclosing HBB from the four predicted corners. This is
            # representation-invariant (including small negative angles) and
            # contains no validation-fitted parameter.
            points = corners[index]
            x1 = min(max(float(points[:, 0].min()), 0.0), 1024.0)
            y1 = min(max(float(points[:, 1].min()), 0.0), 1024.0)
            x2 = min(max(float(points[:, 0].max()), 0.0), 1024.0)
            y2 = min(max(float(points[:, 1].max()), 0.0), 1024.0)
            if x2 <= x1 or y2 <= y1:
                continue
            predictions.append(
                {
                    "id": len(predictions) + 1,
                    "image_id": id_by_stem[stem],
                    "category_id": 0,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(scores[index]),
                }
            )
    if seen != set(id_by_stem):
        raise RuntimeError(f"Incomplete validation coverage: {len(seen)}/{len(id_by_stem)}")
    return predictions


def main() -> None:
    args = parse_args()
    for path in (args.weights, args.data, args.val_list, args.annotations):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.report.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    if getattr(model, "task", None) != "obb":
        raise RuntimeError(f"Expected OBB checkpoint, got task={model.task}")
    parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
    preflight = {
        "component": "dota_obb_specialist_fold0",
        "checkpoint_sha256": sha256(args.weights),
        "parameters": parameter_count,
        "task": model.task,
        "train_images": 1183,
        "valid_images": 290,
        "train_valid_overlap": 0,
        "competition_test_images_read": 0,
    }
    (args.report / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    print(json.dumps(preflight), flush=True)
    if args.preflight_only:
        return

    if args.export_only:
        best = args.weights
    else:
        run_name = "yolo26x_obb_dota_fold0"
        model.train(
            data=str(args.data),
            project=str(args.project),
            name=run_name,
            exist_ok=False,
            epochs=50,
            patience=10,
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
            val=True,
            iou=0.7,
            max_det=20,
            save=True,
            save_period=1,
            plots=False,
            verbose=True,
        )
        run_dir = args.project / run_name
        best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(best)
    best_model = model if args.export_only else YOLO(str(best))
    predictions = export_predictions(best_model, args.val_list, args.annotations)
    prediction_path = args.report / "best_predictions.json"
    prediction_path.write_text(json.dumps({"predictions": predictions}, indent=2) + "\n")
    metrics = evaluate(args.annotations, predictions)
    report = {
        "component": "dota_obb_specialist_fold0",
        "selected_checkpoint": str(best),
        "selected_checkpoint_sha256": sha256(best),
        "parameters": parameter_count,
        "epochs_configured": 50,
        "fixed_hbb": metrics,
        "prediction_rows": len(predictions),
        "competition_test_images_read": 0,
    }
    (args.report / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
