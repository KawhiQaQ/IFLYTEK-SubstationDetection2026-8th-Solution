# Results

## Leaderboard

| Stage | Metric | Score | Rank |
| --- | --- | ---: | ---: |
| Preliminary round | mAP@[0.5:0.95] | 0.92984 | 4 |
| Final round | mAP@[0.5:0.95] | **0.90598** | **8** |

## Fixed validation evidence

The final DOTA-pretrained OBB specialist achieved `0.8649249464` as a standalone horizontal-box detector on fold 0. When inserted into the protected ensemble, the complete fold-0 system reached `0.9369830294`, a gain of `0.0018108642` over the protected parent while preserving its rank-1 prediction.

The same addition improved the final-round leaderboard from `0.90509` to `0.90598` (`+0.00089`). This supports the intended interpretation: the OBB detector is weaker as a standalone model but contributes useful cross-framework localization errors as a low-coverage shadow specialist.

## Full-data replay

The OBB specialist was restarted from the same public DOTA checkpoint and trained on all 1,473 official labeled images for the fold-selected 27 completed epochs. The full run did not load the fold checkpoint and did not use training-set AP to choose an epoch.

The deployed six-checkpoint ensemble totals `568,789,683` bytes. Both competition test sets were excluded from training, pseudo-labeling, validation, and parameter selection.
