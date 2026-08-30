# Method

## 1. Problem formulation

Each sample is a 1024×1024 three-channel high-resolution satellite image. The task is single-class horizontal object detection: for image $I$, the detector predicts a ranked set

$$
\mathcal{D}(I)=\{(b_i,s_i)\}_{i=1}^{K},\qquad
b_i=(x_i^{1},y_i^{1},x_i^{2},y_i^{2}),\qquad K\leq20,
$$

where $b_i$ is an axis-aligned substation box and $s_i\in[0,1]$ is its confidence. The competition metric is COCO-style mAP averaged over ten IoU thresholds from 0.50 to 0.95. This metric gives high weight to precise boundaries: a visually correct detection can still lose substantial score if one edge is displaced.

The final solution therefore separates three responsibilities:

1. **semantic recognition**, handled by an RF-DETR main detector;
2. **high-IoU localization**, handled by learned edge refiners and coordinate triangulation;
3. **recall under domain shift**, handled by heterogeneous lower-ranked specialists.

The ensemble is deliberately asymmetric. The semantic detector owns the leading prediction, while specialists may correct its geometry or add a conservative shadow candidate. This avoids the common failure mode in which uniform averaging improves a weak member but damages a strong one.

## 2. System overview

The deployed system contains six checkpoints:

- a semantic RF-DETR main detector;
- a YOLO26x horizontal detector used as the boundary encoder and proposal source;
- a lightweight four-edge boundary refiner;
- a DEIMv2 coordinate specialist;
- a domain-generalized DEIMv2 shadow specialist;
- a DOTA-pretrained YOLO26x-OBB shadow specialist.

Inference proceeds sequentially. RF-DETR first generates the ordered top-20 parent predictions. The leading parent box is refined using the YOLO boundary branch and triangulated with the coordinate specialist. Two additional specialists are then evaluated independently. A specialist candidate is admitted only when it is geometrically different from the protected parent, receives a bounded confidence, and survives the final stable top-20 sort.

All thresholds and coefficients below are fixed in the released code. Test images are never used to fit them.

## 3. Semantic RF-DETR main detector

### 3.1 Remote-sensing semantic initialization

The main detector uses RF-DETR-Large at 1024 resolution. Its representation is adapted toward satellite imagery through a DINOv3 satellite semantic task vector. Let $\theta_0$ denote the common detector initialization and $\theta_{\mathrm{sat}}$ the satellite-adapted representation. The released semantic initialization follows

$$
\theta_{\mathrm{sem}}
=\theta_0+\alpha(\theta_{\mathrm{sat}}-\theta_0),
\qquad \alpha=0.25.
$$

This preserves most of the generic detector while importing remote-sensing texture and layout priors. The complete detector—not only a small prediction head—is then optimized on official annotations.

### 3.2 Boundary-evidence head

DETR queries are strong at object-level reasoning but their coarse feature path can be less sensitive to individual fence or equipment-yard edges. For each query box, the boundary-evidence head samples high-resolution feature profiles on the four sides. Each side uses points along the tangent direction and offsets along the outward-to-inward normal direction. The sampled local evidence is fused with the decoder query and box geometry to predict a normalized edge shift

$$
\Delta_i=(\delta_l,\delta_t,\delta_r,\delta_b)_i.
$$

For a reference box with width $w_i$ and height $h_i$, the target is the scale-normalized difference between reference and ground-truth edges:

$$
\Delta_i^{*}
=\mathrm{clip}\left(
\frac{b_i^{*}-b_i}{(w_i,h_i,w_i,h_i)},
-\rho,\rho
\right).
$$

The additional edge objective is a normalized Smooth-L1 loss,

$$
\mathcal{L}_{\mathrm{edge}}
=\frac{1}{N}\sum_i
\mathrm{SmoothL1}_{\beta}(\Delta_i,\Delta_i^{*}),
$$

and is added to the original classification, L1-box, GIoU, and denoising objectives. The final linear layer is initialized to zero, so adding the module initially leaves the parent detector unchanged.

### 3.3 Edge-wise query consensus

A lower-ranked query can contain a better left or bottom edge than the highest-confidence query. To exploit this without averaging unrelated objects, each seed query collects the five highest-confidence proposals from the same Group-DETR partition whose overlap exceeds 0.50. Four attention distributions are learned independently—one for each edge. Query features, pairwise box geometry, confidence, and overlap cues produce a consensus residual around the boundary-aware reference.

For edge $e\in\{l,t,r,b\}$, the aggregate can be written as

$$
z_i^{e}=\sum_{j\in\mathcal{N}(i)}a_{ij}^{e}v_j^{e},
\qquad
\sum_j a_{ij}^{e}=1,
$$

followed by a bounded residual $\widehat{\delta}_i^{e}$. Neighbour selection is detached; gradient learning occurs inside the attention and residual projections. A second Smooth-L1 edge objective directly supervises the final consensus box. Like the boundary head, the consensus regressor is zero-initialized to make installation an exact identity before training.

## 4. YOLO boundary-distribution refiner

The auxiliary YOLO26x detector provides complementary convolutional edge evidence. For every YOLO proposal, the refiner extracts a tight crop and a wider context crop. YOLO layers 0–4 are frozen and shared between the two views; stride-4 and stride-8 features are projected, fused, and combined with proposal width, height, and aspect-ratio features.

Instead of regressing an unconstrained scalar, each side predicts a categorical distribution over 33 non-uniform residual bins. The grid is denser around zero, which concentrates resolution where most high-IoU corrections occur. If $p_{e,k}$ is the probability of bin $r_k$, the decoded side residual is

$$
\widehat r_e=\sum_{k=1}^{33}p_{e,k}r_k.
$$

Training uses two-hot distribution supervision together with entropy, aligned-IoU, and Smooth-L1 terms:

$$
\mathcal{L}_{\mathrm{ref}}
=\mathcal{L}_{\mathrm{2hot}}
+0.10\,\mathcal{H}(p)
+\mathcal{L}_{\mathrm{IoU}}
+2\,\mathcal{L}_{\mathrm{res}}.
$$

The shallow YOLO encoder remains frozen; only the refinement module is saved in its small checkpoint. Zero-initialized classification logits decode to zero expected displacement, providing a safe identity initialization.

At system inference, the boundary-refined box $b_r$ is matched to the semantic parent box $b_1$ using IoU. A match below 0.50 falls back to $b_1$. The accepted correction is deliberately damped:

$$
b_p=b_1+\lambda(b_r-b_1),
\qquad \lambda=0.24456708133220673.
$$

The confidence and rank of the semantic prediction are not changed.

## 5. DEIMv2 localization specialists

### 5.1 Coordinate specialist and robust triangulation

The first DEIMv2 detector is initialized from a DINOv3-based public detector and adapted at 1024 resolution. Its architecture and matching dynamics differ from both RF-DETR and YOLO, producing useful independent edge errors.

Let $b_c$ be the coordinate specialist's rank-1 box. The final protected parent geometry is the coordinate-wise median

$$
b_{\mathrm{rank1}}
=\mathrm{median}(b_p,b_r,b_c).
$$

The median is taken independently for $(x^1,y^1,x^2,y^2)$. It behaves like a robust three-estimator consensus: one extreme side prediction cannot pull the result arbitrarily far. If the median box is degenerate, the system falls back to $b_p$. The semantic parent's confidence is retained.

### 5.2 Query-conditioned high-resolution boundary path

The domain specialist augments DEIMv2 with a local boundary decoder. It captures the stride-4 spatial-prior feature, the encoded stride-8 feature, and the final decoder query. For each predicted side, fixed outside/boundary/inside strips are sampled from the two feature maps. The local strips, global query, confidence, edge identity, and box geometry predict a bounded relative correction. The analytic conversion from four edge residuals to $(c_x,c_y,w,h)$ guarantees positive sizes within the configured residual range.

### 5.3 Train-only semantic style mixing

To improve robustness to satellite sensor, season, tone, and compression changes, only the deepest semantic feature is style-mixed during training. For feature $f$, per-channel mean and deviation are

$$
\mu(f)=\mathbb{E}_{h,w}[f],\qquad
\sigma(f)=\sqrt{\mathrm{Var}_{h,w}(f)+\epsilon}.
$$

Statistics from a different image in the mini-batch are mixed with coefficient $\gamma\sim\mathrm{Beta}(0.1,0.1)$:

$$
\widetilde f
=\frac{f-\mu(f)}{\sigma(f)}
\left[\gamma\sigma(f)+(1-\gamma)\sigma(f')\right]
+\gamma\mu(f)+(1-\gamma)\mu(f').
$$

The transformation is applied with probability 0.5 and is an exact identity during validation and deployment. Crucially, the stride-4 edge path is not stylized, separating semantic domain randomization from precise boundary evidence.

## 6. DOTA-pretrained oriented specialist

Substations are often rectangular compounds with long, coherent boundaries. The final specialist starts from the public DOTA-pretrained `yolo26x-obb.pt` checkpoint and fine-tunes the complete backbone, neck, assignment, box, and angle branches. Original horizontal labels are represented as zero-angle four-corner OBB polygons during training; no synthetic rotation angle is inferred from the label.

At inference, the specialist predicts oriented corners $\{(x_j,y_j)\}_{j=1}^{4}$. The competition requires horizontal boxes, so the conversion is deterministic:

$$
b_{\mathrm{HBB}}
=\left(
\min_jx_j,\min_jy_j,\max_jx_j,\max_jy_j
\right).
$$

This conversion has no fitted parameter. The specialist's fold-0 checkpoint was selected using the official COCO-style horizontal-box metric, and the full-data model was retrained from the same public DOTA initialization for the selected 27 completed epochs.

## 7. Protected shadow-candidate fusion

After rank-1 triangulation, the domain specialist and the OBB specialist are added sequentially. For a specialist candidate $(b_s,s_s)$, define

$$
u=\mathrm{IoU}(b_{\mathrm{rank1}},b_s).
$$

If $u\geq0.95$, the candidate is redundant and discarded. Otherwise its confidence is bounded using the parent's first and second scores:

$$
s_{\mathrm{shadow}}
=s_2\min(s_1,s_s).
$$

The candidate is appended but is unable to outrank the protected parent. After each insertion, candidates are stably sorted by descending score; parent candidates win deterministic ties, and only the first 20 rows are retained. The same fixed rule is used for both shadow specialists.

This procedure differs from WBF or score-optimized stacking in three ways: it never alters parent rank-1, it uses a geometric admission condition rather than fitted ensemble weights, and it preserves calibration by bounding every specialist score below the parent's second score.

## 8. Training and compliance principles

The official fold uses 1,183 training and 290 validation images with zero overlap. Model selection uses mAP@[0.5:0.95], with AP at 0.95 tracked as a localization diagnostic. Once a component's completed epoch count is selected, full training uses all 1,473 official labeled images from the same public initialization and does not use training-set AP for checkpoint selection.

The preliminary-round and final-round test sets are read only by the inference entry point. They are not used for training, pseudo-labeling, threshold search, score calibration, or early stopping. The released checkpoint manifest fixes every filename, byte size, and SHA-256 digest. Total deployed model size is `568,789,683` bytes, strictly below the `600,000,000`-byte competition limit.

## 9. Ablation summary

The fixed fold-0 complete ensemble reaches **0.9369830 mAP@[0.5:0.95]**. Adding the DOTA-OBB shadow branch to its protected parent improved the fixed fold-0 system by **0.0018109** and the final-round leaderboard by **0.00089**, from 0.90509 to **0.90598**. These gains were obtained without public-leaderboard-fitted thresholds, WBF coefficients, or test-time label access.
