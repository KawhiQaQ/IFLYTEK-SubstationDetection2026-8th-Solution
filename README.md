<div align="center">

<h1>IFLYTEK SubstationDetection2026: 8th-Place Solution</h1>

<p><strong>High-Resolution Remote-Sensing Substation Detection Challenge</strong></p>

<p>
  <a href="README.md">English</a> |
  <a href="README_CN.md">简体中文</a>
</p>

<p>
  <img alt="Python 3.11" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch 2.5.1" src="https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Final rank 8th" src="https://img.shields.io/badge/Final_Rank-8th-6F42C1">
  <img alt="Final mAP 0.90598" src="https://img.shields.io/badge/Final_mAP-0.90598-2EA44F">
  <img alt="License AGPL-3.0" src="https://img.shields.io/badge/License-AGPL--3.0-blue">
</p>

</div>

---

[Method](docs/METHOD.md) | [Reproduction](docs/REPRODUCTION.md) | [Weights](weights/README.md)

This repository contains the **8th-place final-round solution** to the iFLYTEK High-Resolution Remote-Sensing Substation Detection Challenge, developed by team **kawhi00**. The detector combines a semantic RF-DETR main model, a four-edge boundary refiner, two heterogeneous DEIMv2 specialists, and a DOTA-pretrained YOLO26x-OBB specialist.

The final-round leaderboard score is **0.90598 mAP@[0.5:0.95]**. The six released checkpoint files occupy `568,789,683` bytes in total, satisfying the competition's 600 MB model-weight limit.

## Results

| Stage | Metric | Score | Rank |
| --- | --- | ---: | ---: |
| Preliminary round | mAP@[0.5:0.95] | 0.92984 | 4th |
| Final round | mAP@[0.5:0.95] | **0.90598** | **8th** |

## Method at a glance

The system is a protected candidate ensemble, not a uniform weighted average:

1. RF-DETR produces the ordered semantic prediction set.
2. A frozen-YOLO boundary refiner adjusts the leading box without changing its confidence.
3. A DEIMv2 coordinate specialist participates in coordinate-wise rank-1 triangulation.
4. A domain-generalized DEIMv2 detector and a DOTA-OBB detector contribute only geometrically complementary shadow candidates.
5. The semantic parent's rank-1 prediction is always protected; the output is stably sorted and truncated to 20 boxes.

## Quick start

```bash
conda env create -f environment.yml
conda activate substation-detection

# Download the six released checkpoints into weights/ first.
python scripts/verify_weights.py --weights weights

python scripts/infer.py \
  --images /path/to/test/images \
  --weights weights \
  --output outputs/predictions \
  --device cuda:0 \
  --batch-size 2
```

Each image produces one UTF-8 `.txt` file with normalized YOLO detections:

```text
class_id x_center y_center width height confidence
```

The destination must not already contain prediction files. This prevents accidental mixing of outputs from different runs.

## Repository layout

```text
.
├── src/
│   ├── infer_parent_ensemble.py       # semantic main + boundary triangulation
│   ├── infer_protected_ensemble.py    # domain-specialist shadow insertion
│   ├── infer_final_ensemble.py        # final OBB-specialist ensemble
│   ├── rfdetr/                        # pinned RF-DETR source
│   ├── external/DEIMv2/               # pinned DEIMv2 source
│   └── workspace/                     # custom model modules and configs
├── training/                          # OBB preparation, training, and audit tools
├── scripts/                           # public inference and checkpoint verification
├── weights/                           # manifest; binary checkpoints are not in Git
├── docs/                              # method, data, training, and reproduction notes
└── tests/                             # CPU-only repository contract tests
```

## Reproduction and training

- Use [docs/REPRODUCTION.md](docs/REPRODUCTION.md) either to reproduce the released score from the exact checkpoints or to train every component yourself.
- Download the released checkpoints from [Baidu Netdisk](https://pan.baidu.com/s/10u1sjJgp585YQ6dMdfmtYg?pwd=d4hh) (extraction code: `d4hh`). Checkpoint filenames, byte sizes, and SHA-256 digests are recorded in [weights/manifest.json](weights/manifest.json).

## License

The project is released under AGPL-3.0-only because the final system depends on AGPL-licensed Ultralytics software. Vendored RF-DETR and DEIMv2 code retain their Apache-2.0 licenses. Dataset and checkpoint use remains subject to the original providers' terms.
