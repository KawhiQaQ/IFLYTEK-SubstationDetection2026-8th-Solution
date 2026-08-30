# Reproduction

This repository supports two reproducibility goals:

1. **Result reproduction:** run the released six-checkpoint ensemble and reproduce the final submission format and inference graph that achieved 0.90598.
2. **Clean retraining:** start from public model initializations, train the individual model families on official labels, select epochs on a leakage-free fold, retrain on all labels, and rebuild the protected ensemble.

The first route is deterministic apart from small CUDA-kernel differences and is the recommended way to verify the published result. The second route is intended for research and adaptation; stochastic training, upstream package changes, and a different train-validation split can produce a different score.

## 1. Environment

Recommended hardware and software:

- Linux x86_64;
- Python 3.10 or 3.11;
- NVIDIA GPU with at least 16 GB VRAM; 24 GB is recommended;
- PyTorch 2.5.1, torchvision 0.20.1, CUDA 12.1.

Create the pinned environment:

```bash
conda env create -f environment.yml
conda activate substation-detection
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The deployment platform used a separate Python 3.8 compatibility bundle. That bundle is unnecessary for normal local inference and is therefore not committed.

## 2. Obtain the released checkpoints

Download the weight archive and place these files directly in `weights/`:

```text
weights/
├── rfdetr_semantic_main.pth
├── yolo26x_hbb_detector.pt
├── yolo_boundary_refiner.pt
├── deimv2_coordinate_specialist_fp16.pth
├── deimv2_domain_generalized_specialist_int8.pth
└── yolo26x_obb_dota_specialist.pt
```

Download: [Baidu Netdisk](https://pan.baidu.com/s/10u1sjJgp585YQ6dMdfmtYg?pwd=d4hh), extraction code: `d4hh`.

Verify the archive before inference:

```bash
python scripts/verify_weights.py --weights weights
```

The verifier checks that the required six files are present and that their total model size stays below the competition limit. The authoritative file identities are in `weights/manifest.json`.

## 3. Prepare images

The inference entry point accepts `.jpg`, `.jpeg`, or `.png` files. Competition reproduction expects 1024×1024 images with unique stems:

```text
data/test/images/
├── 000001.jpg
├── 000002.jpg
└── ...
```

Do not place labels in this directory. The inference program never opens a label file.

## 4. Run the released ensemble

```bash
python scripts/infer.py \
  --images data/test/images \
  --weights weights \
  --output outputs/final_predictions \
  --device cuda:0 \
  --batch-size 2
```

Optional `--expected-images N` adds a strict image-count check. Use a new output directory for every run; existing `.txt` files cause an error instead of being silently mixed with new predictions.

The output for each image is:

```text
0 x_center y_center width height confidence
```

Coordinates are normalized to `[0,1]`, there is no header, and at most 20 rows are emitted per image.

## 5. Sanity-check the generated submission

For every output file, check the following:

- exactly six numeric columns per non-empty line;
- class id equals zero;
- normalized box coordinates and confidence lie in `[0,1]`;
- width and height are positive;
- no more than 20 rows per image;
- output stems exactly match input stems.

The inference source also enforces the structural contracts that materially affected the competition result: semantic rank-1 protection, fixed geometric admission at IoU 0.95, stable tie ordering, deterministic OBB-to-HBB conversion, and a strict 600 MB checkpoint limit.

## 6. Train the models yourself

This section gives the complete clean-training route for the six-component ensemble. The released checkpoints remain the reference for the published leaderboard score. A clean retrain is scientifically equivalent but will not be byte-identical because GPU kernels, random augmentation order, public checkpoint revisions, and fold construction affect optimization.

### 6.1 Public initializations

| Component | Initialization | Source |
| --- | --- | --- |
| Semantic main | RF-DETR-Large | [RF-DETR](https://github.com/roboflow/rf-detr) |
| Satellite representation | DINOv3 satellite model | [DINOv3](https://github.com/facebookresearch/dinov3) |
| HBB detector and boundary encoder | YOLO26x general checkpoint | [Ultralytics](https://github.com/ultralytics/ultralytics) |
| Coordinate specialists | DEIMv2 DINOv3-X detector | [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) |
| Oriented specialist | YOLO26x-OBB pretrained on DOTA | [Ultralytics](https://github.com/ultralytics/ultralytics) / [DOTA](https://captain-whu.github.io/DOTA/) |

Some DINOv3 files are gated by their provider. Accept the corresponding license before downloading them. Keep public initializations under `pretrained/`; they are not committed to this repository.

### 6.2 Official data layout

Arrange the 1,473 labeled images as standard YOLO data:

```text
data/official/
├── images/
│   ├── image_a.jpg
│   └── ...
├── labels/
│   ├── image_a.txt
│   └── ...
└── data.yaml
```

Each label row is `0 x_center y_center width height`, with normalized coordinates. Before training, audit image-label stem equality, image size, class id, finite values, and positive widths and heights.

### 6.3 Create the development split

The development protocol uses 1,183 images for training and 290 for validation, with zero overlap. If the original split list is unavailable, create a deterministic replacement:

```bash
python training/create_split.py \
  --images data/official/images \
  --output data/splits \
  --seed 3407 \
  --valid-images 290
```

This creates `fold0_train.txt`, `fold0_val.txt`, and `full_train.txt`. A different split is valid for research but its local score is not directly comparable with the reported fold-0 score. Convert the fixed lists to COCO for DEIMv2 evaluation:

```bash
python training/convert_yolo_to_coco.py \
  --train-list data/splits/fold0_train.txt \
  --val-list data/splits/fold0_val.txt \
  --output-dir data/coco
```

RF-DETR expects a Roboflow-style COCO tree. Symlink images into `data/rfdetr/train/` and `data/rfdetr/valid/`, then save the matching COCO files as `_annotations.coco.json` in each directory. Symlinks avoid duplicating the 1024×1024 images.

### 6.4 Train the semantic RF-DETR main model

The semantic detector uses RF-DETR-Large at 1024 resolution, the boundary-evidence module, and edge-wise query consensus. The training patch is in `src/workspace/models/rfdetr_query_consensus.py`; architecture values are in `src/workspace/configs/semantic_main.yaml`.

```bash
PYTHONPATH=src python - <<'PY'
import yaml
from pathlib import Path
from rfdetr import RFDETRLarge
from infer_parent_ensemble import query_config
from workspace.models.rfdetr_query_consensus import configure_training_patch

recipe = yaml.safe_load(Path("src/workspace/configs/semantic_main.yaml").read_text())
configure_training_patch(query_config(recipe), bbox_only_evaluation=True)

model = RFDETRLarge(
    pretrain_weights="pretrained/rf-detr-large.pth",
    resolution=1024,
    num_classes=1,
)
model.train(
    dataset_dir="data/rfdetr",
    output_dir="outputs/cv/semantic_main",
    epochs=36,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    lr_encoder=1.5e-4,
    weight_decay=1e-4,
    use_ema=True,
    ema_decay=0.993,
    ema_tau=100,
    multi_scale=True,
    expanded_scales=True,
    seed=3407,
)
PY
```

If a licensed satellite-adapted RF-DETR-compatible checkpoint is available, apply the task vector described in [METHOD.md](METHOD.md) before target training. Otherwise the public RF-DETR initialization is a valid clean baseline, but it is not expected to reproduce the exact released semantic checkpoint.

Select the completed epoch count by validation mAP@[0.5:0.95], while also monitoring AP at 0.95. For full-data training, restart from the same public initialization, use all 1,473 official images, and train for the frozen number of epochs. Do not select a full-data checkpoint using training-set AP.

### 6.5 Train the YOLO HBB detector and edge refiner

Train the single-class YOLO26x HBB detector on the same split:

```bash
yolo detect train \
  model=pretrained/yolo26x.pt \
  data=data/official/data.yaml \
  imgsz=1024 batch=2 epochs=50 patience=10 \
  optimizer=AdamW lr0=0.0001 cos_lr=True \
  single_cls=True max_det=20 seed=3407
```

Run the trained detector on the **training split only** with `conf=0.001`, `iou=0.7`, and `max_det=20`, saving image paths, `xyxy` proposals, and confidences. Train `BoundaryDistributionRefiner` from `src/workspace/models/boundary_refiner.py` on proposal/ground-truth matches with IoU at least 0.50. Utilities in `src/workspace/models/refinement_data.py` build tight and wide crops and normalized edge targets.

```python
output = refiner(tight_crop, wide_crop, proposal_geometry)
loss, metrics = refinement_loss(refiner, output, target_edge_residuals)
loss.backward()
```

Freeze YOLO layers 0–4 during refiner training and serialize only `refiner.trainable_state_dict()`. Fit the boundary blend coefficient exclusively from train-only proposals; never fit it on either competition test set. The released inference coefficient is fixed at `0.24456708133220673`.

For full-data replay, restart the HBB detector from the same public initialization, generate proposals for all 1,473 labeled images, and retrain the refiner on those proposals.

### 6.6 Train the DEIMv2 specialists

Train the coordinate specialist using the vendored DEIMv2 entry point:

```bash
python training/train_deim_specialist.py \
  --config src/workspace/configs/coordinate_specialist_fold0.yml \
  --tuning pretrained/deimv2_dinov3_x_coco.pth \
  --use-amp --seed 3407
```

Train the query-conditioned domain-generalized specialist from the same compatible public detector family:

```bash
python training/train_deim_specialist.py \
  --config src/workspace/configs/domain_specialist_fold0.yml \
  --tuning pretrained/deimv2_dinov3_x_coco.pth \
  --use-amp --seed 3407
```

Both models optimize the complete detector. The domain specialist applies semantic style mixing only during training; validation and inference remain deterministic. Select completed epoch counts on fold 0, update the full-data configs to use all official annotations, and restart from the same public initializations.

The released domain checkpoint stores a fixed subset of large two-dimensional tensors with per-output-row symmetric INT8 storage. `src/infer_protected_ensemble.py` restores them to FP16 before model construction. This is storage compression for the 600 MB rule, not quantized forward inference.

### 6.7 Train the DOTA-OBB specialist

Convert the official horizontal labels to zero-angle OBB rectangles:

```bash
python training/prepare_obb_dataset.py \
  --train-list data/splits/fold0_train.txt \
  --val-list data/splits/fold0_val.txt \
  --output data/obb_fold0
```

Train and export strict enclosing-HBB validation predictions:

```bash
python training/train_obb_fold0.py \
  --weights pretrained/yolo26x-obb.pt \
  --data data/obb_fold0/dataset.yaml \
  --val-list data/splits/fold0_val.txt \
  --annotations data/coco/fold0_val.json \
  --project outputs/cv \
  --report reports/obb_fold0
```

The recipe uses input size 1024, batch 1, AdamW with learning rate `1e-4`, D4 flips, light photometric and scale augmentation, at most 50 epochs, and patience 10. The released run selected 27 completed epochs.

Build the train-only full-data view and restart from the same public DOTA checkpoint:

```bash
python training/prepare_obb_dataset.py \
  --train-list data/splits/full_train.txt \
  --val-list data/splits/fold0_val.txt \
  --output data/obb_full \
  --train-only

python training/train_obb_full.py \
  --weights pretrained/yolo26x-obb.pt \
  --data data/obb_full/dataset.yaml \
  --project outputs/full \
  --report reports/obb_full
```

The full run lasts exactly 27 completed epochs and does not evaluate its training set for model selection.

### 6.8 Audit protected fusion

Export fold-0 COCO predictions for the protected parent and OBB specialist, then run:

```bash
python training/evaluate_protected_shadow.py \
  --annotations data/coco/fold0_val.json \
  --parent predictions/protected_parent.json \
  --specialist reports/obb_fold0/best_predictions.json \
  --output reports/obb_shadow_audit \
  --reference 0.9351721652078016
```

The audit fixes mutual IoU at 0.95, preserves parent rank-1, uses the bounded shadow score, retains top-20, and reports the exact mAP difference without searching ensemble weights.

### 6.9 Assemble the full-data checkpoints

Copy the six selected full-data checkpoints into `weights/` using the semantic filenames in `weights/manifest.json`, then verify and infer:

```bash
python scripts/verify_weights.py --weights weights
python scripts/infer.py \
  --images data/test/images \
  --weights weights \
  --output outputs/final_predictions \
  --device cuda:0 --batch-size 2
```

If you trained your own weights, their file identities will differ from the released manifest. Keep a separate manifest instead of replacing the published reference record.

## 7. Repository checks

Before a long run, execute:

```bash
python -m unittest discover -s tests -v
python -m compileall -q scripts training src
```

These CPU checks catch missing files, stale public names, invalid manifests, and syntax drift. A final CUDA end-to-end run on several 1024×1024 images is still required before generating a competition submission.
