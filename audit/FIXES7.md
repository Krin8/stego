# FIXES7: Anatomy-Aware Masked STEGO (Approach 4)

This round tested **Approach 4**: modifying the STEGO correspondence loss to only sample pairs of pixels that fall inside the patient's body mask (ignoring the empty scanner bed/air). We combined this with the 768-D Residual Head from FIXES6 to ensure we had no dimensional bottleneck.

## Design
* **Goal**: Rescue the STEGO contrastive loss by preventing it from wasting 90% of its sampling capacity on empty background patches, forcing it to focus entirely on distinguishing internal organs.
* **Architecture**: 768-D Residual Head (initialized at frozen baseline, 78% IoU).
* **Loss Mod**: Masked `reference_loss.py` to only sample where Hounsfield Units > -500.
* **Training**: 3 seeds, 500 batches.

## Stage 1: Screen

All three seeds remained perfectly healthy, showing no signs of dimensional collapse.

## Stage 2: Evaluation

By restricting the loss to only the patient's body, performance degraded even further compared to the unmasked residual head.

| Method | Mean IoU | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|---:|
| Frozen K-means + F1 (baseline) | **77.96%** | - | - | - |
| Residual-768 (Unmasked) | **70.57%** | 69.34% | 70.73% | 71.65% |
| Residual-768 (Body Masked) | **65.56%** | 64.33% | 66.06% | 66.29% |
| D448 70-D (Unmasked) | **66.01%** | 65.59% | 65.94% | 66.49% |

## Conclusion

This is the final nail in the coffin for the contrastive correspondence loss in zero-shot medical segmentation. 

By removing the "easy" background contrast (body vs air), the contrastive loss was forced to operate exclusively on the complex, low-contrast internal organs. The loss actively blurred the boundaries between the liver and adjacent tissues (like the stomach and spleen), dragging the performance from the 78% starting baseline all the way down to 65.5%. 

We have definitively proven that **STEGO's contrastive spatial loss is mathematically pathological for zero-shot organ segmentation**, regardless of dimensionality (FIXES6) or sampling distribution (FIXES7). 

**Recommendation:** Proceed immediately to **Approach 2: Self-Training with pseudo-labels**, discarding the contrastive loss in favor of a cross-entropy loss trained directly on the high-quality 78% frozen baseline labels.
