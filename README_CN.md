<div align="center">

<h1>IFLYTEK SubstationDetection2026：第 8 名方案</h1>

<p><strong>讯飞AI算法赛高分辨率遥感影像变电站识别挑战赛</strong></p>

<p>
  <a href="README.md">English</a> |
  <a href="README_CN.md">简体中文</a>
</p>

<p>
  <img alt="Python 3.11" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch 2.5.1" src="https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="复赛第8名" src="https://img.shields.io/badge/Final_Rank-8th-6F42C1">
  <img alt="复赛mAP 0.90598" src="https://img.shields.io/badge/Final_mAP-0.90598-2EA44F">
  <img alt="AGPL-3.0许可证" src="https://img.shields.io/badge/License-AGPL--3.0-blue">
</p>

</div>

---

[方法说明](docs/METHOD.md) | [复现](docs/REPRODUCTION.md) | [模型权重](weights/README.md)

本仓库公开队伍 **kawhi00** 在讯飞AI算法赛“高分辨率遥感影像变电站识别挑战赛”中的**复赛第 8 名方案**。系统由 RF-DETR 语义主模型、四边界细化器、两个异构 DEIMv2 specialist，以及一个经 DOTA 预训练的 YOLO26x-OBB specialist 组成。

复赛成绩为 **0.90598 mAP@[0.5:0.95]**。六个部署权重共 `568,789,683` bytes，满足比赛模型权重不超过 600 MB 的限制。

## 成绩

| 阶段 | 指标 | 分数 | 排名 |
| --- | --- | ---: | ---: |
| 初赛 | mAP@[0.5:0.95] | 0.92984 | 第 4 名 |
| 复赛 | mAP@[0.5:0.95] | **0.90598** | **第 8 名** |

## 方法概览

最终方案不是简单的多模型等权平均，而是一个保护主预测的候选级集成系统：

1. RF-DETR 输出按置信度排列的语义检测结果；
2. 使用冻结的 YOLO 浅层特征和轻量边界头细化 rank-1 框，同时保留原置信度；
3. 由 DEIMv2 坐标 specialist 与主模型进行逐坐标中位数三角化；
4. 域泛化 DEIMv2 和 DOTA-OBB specialist 只在与主框存在足够几何分歧时提供 shadow candidate；
5. 主系统 rank-1 始终受保护，所有候选稳定排序后固定保留 top-20。

## 快速开始

```bash
conda env create -f environment.yml
conda activate substation-detection

# 先将六个公开权重放入 weights/。
python scripts/verify_weights.py --weights weights

python scripts/infer.py \
  --images /path/to/test/images \
  --weights weights \
  --output outputs/predictions \
  --device cuda:0 \
  --batch-size 2
```

每张图像会生成一个同名 UTF-8 `.txt` 文件，每行格式为：

```text
class_id x_center y_center width height confidence
```

坐标均归一化到 `[0,1]`，每张图最多保留 20 个候选框。输出目录不能已有 `.txt` 文件，避免不同推理结果意外混合。

## 仓库结构

```text
.
├── src/
│   ├── infer_parent_ensemble.py       # 语义主模型、边界细化与坐标三角化
│   ├── infer_protected_ensemble.py    # 域泛化 specialist 的受保护候选插入
│   ├── infer_final_ensemble.py        # 最终 OBB specialist 集成
│   ├── rfdetr/                        # 固定版本 RF-DETR 源码
│   ├── external/DEIMv2/               # 固定版本 DEIMv2 源码
│   └── workspace/                     # 自定义模型模块与配置
├── training/                          # OBB 数据准备、训练和审计脚本
├── scripts/                           # 推理与权重校验入口
├── weights/                           # 权重清单；二进制权重不进入 Git
├── docs/                              # 方法、数据、训练和复现文档
└── tests/                             # 无需 GPU 的仓库契约测试
```

## 复现与自行训练

- [docs/REPRODUCTION.md](docs/REPRODUCTION.md) 同时包含两条完整路线：使用精确权重复现公开成绩，以及从公共预训练权重开始自行训练全部组件；
- 六个权重可从[百度网盘](https://pan.baidu.com/s/10u1sjJgp585YQ6dMdfmtYg?pwd=d4hh)下载，提取码：`d4hh`。公开文件名、大小与 SHA-256 记录在 [weights/manifest.json](weights/manifest.json)。

## 开源许可

本项目按 AGPL-3.0-only 发布，因为最终系统依赖 AGPL-3.0 的 Ultralytics。仓库内 RF-DETR 与 DEIMv2 代码保留各自的 Apache-2.0 许可。数据集和权重仍受原始提供方条款约束。
