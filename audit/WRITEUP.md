# Comprehensive Research Handoff: Label-Free CT Liver Segmentation

This document serves as the complete technical record of our research journey to achieve label-free (zero-shot) CT liver segmentation on the CHAOS dataset using BiomedCLIP features. 

---

## Part 1: The Research Journey & Experiments

Our goal was label-free CT liver segmentation compared against the frozen features.

### 1. The Baseline (Approach 3)
* **What we did:** We extracted the 768-D dense features from the frozen BiomedCLIP vision encoder and applied K-means clustering (k=8) directly to the features. 
* **The Result:** 
  * Hard assignment (Raw K-means): **70.83% IoU**
  * Inductive F1 Refinement (Guided filter, hole fill, closing): **77.96% IoU**
  * Transductive F1 Refinement: **78.8% – 80.2% IoU**
* **Finding:** The frozen BiomedCLIP representations are incredibly rich and encode semantic boundaries remarkably well. Note that cluster IDs are arbitrary per fit (e.g., Seed 0 mapped cluster 2 to the liver, while Seeds 1 and 2 mapped clusters 3 and 4).

### 2. Experiment 1: The Dimensional Bottleneck (Approach 1)
* **What we did:** The original STEGO implementation projected the 768-D features down to 70 dimensions. We hypothesized this was a dimensional bottleneck destroying critical information. We built a **768-D Residual Head** initialized near identity and trained it using the STEGO contrastive loss.
* **The Result:** The F1-refined performance scored **70.57%**. *(Note: Approach 1 and Approaches 4 & 6 used inductive scoring: a 16-patient teacher with 4 patients scored).*
* **Analysis:** The residual head's step-0 score is still missing. Without it, 70.57 against 77.96 isn't a controlled comparison, because a 'near-identity' head may not reproduce the baseline exactly. Furthermore, PREFLIGHT diagnostics measured representation collapse on the original 70-D head. The loss structure rewards collapse at these pinned shifts (b_self=0.18, b_knn=0.12, b_rand=0.46 from the repository defaults). The loss itself isn't necessarily broken; the hyperparameter calibration is fundamentally imbalanced.

### 3. Experiment 2: Anatomy-Aware Masked Sampling (Approaches 4 & 6)
* **What we did:** We computed exact body masks using Hounsfield Units (-500 HU) and restricted the contrastive loss to sample only within the patient's body.
* **The Result:** The inductive F1-refined performance dropped to **65.56%**.
* **Analysis (Untested Hypothesis):** Restricting sampling exclusively to the body dramatically alters the attraction/repulsion balance. We hypothesize that without re-tuning the spatial shifts, the loss is forced into a highly unstable regime, ultimately blurring boundaries between adjacent internal organs. However, the exact attraction fractions under body-masked sampling have not yet been empirically measured.

### 4. Proposed Next Step: Self-Training (Approaches 2 & 5)
* **What we did:** We used the robust K-means clusters as pseudo-labels to train a Convolutional Neural Network (CNN) via standard Cross-Entropy loss on the original, high-resolution CT image.
* **The Result:** A CPU proof-of-concept (3-layer Mini-UNet, 3 epochs) scored **59.15%** (Global hard IoU on the validation split, using the N3 inductive naming logic). 
* **Analysis:** This score is far below the teacher (70.83% hard, 77.96% F1) and indicates severe underfitting. A full GPU run is required to determine if this architecture's inductive bias can actually surpass the 80.03% inductive target (77.96% + 1 SD).

### 5. Experiment 3: Co-located Adjacent-Slice Loss (P1)
* **What we did:** We replaced the kNN inter-image partner term with a co-located adjacent-slice term (P1): for each anchor slice, draw a partner from {-2, -1, +1, +2} physical positions in the same patient/series and compare only the spatially co-located (diagonal) patch pairs, not all Cartesian pairs. The self and random terms remain unchanged. An amended calibration protocol selected the P1 shift from five attraction targets (20%, 40%, 60%, 80%, 90%) using only label-free full-loss admissibility (constant > random > K8 in the anchor-weighted mean and in every batch). Amendment commit: `11c2008`.
* **P1 calibration result:** Selected target **90%**, shift **0.8869667053222656**. K8 advantage (random minus K8 in full loss): **0.0335785423**. Selection used only label-free full losses. The original 20% target failed the strict gate (K8 > random in full loss); targets 40%–90% all passed, and 90% had the greatest K8 advantage.
* **Screen result:** Both C448 (control) and P1 passed all three-seed 150-batch screens. All six runs remained healthy: code std > 0.113, cross-image cosine < 0.029, all 8 clusters used. Screen evidence: [`screen_summary.md`](file:///Users/navneetbavineni/STEGO/audit/colocated/screen_summary.md).
* **Stage 2 status: COMPLETED.** The mandatory overlap check passed and training ran for 500 batches for all three seeds for both C448 and P1. All runs remained healthy. All predictions were sealed before opening masks. (Note: The earlier permission failure with `ps` was an artifact of the agent's restricted execution sandbox; no code or checks were bypassed to run Stage 2 successfully.)
* **Segmentation criterion: FAILURE.** The target criterion was a mean F1 pooled-pixel IoU (at 224 primary scale) over seeds of at least 80.03%, and no seed below 77.96%. The trained head was the 70-D head, not the 768-D residual head.
  * **C448 (Control):** Mean F1 IoU (224) was **66.10%**. 
    * Seed 0: 65.64% F1 (70.68% hard)
    * Seed 1: 66.96% F1 (70.55% hard)
    * Seed 2: 65.69% F1 (70.81% hard)
  * **P1 (Co-located):** Mean F1 IoU (224) was **67.89%**. 
    * Seed 0: 67.81% F1 (70.87% hard)
    * Seed 1: 66.95% F1 (70.96% hard)
    * Seed 2: 68.93% F1 (70.76% hard)
  * **Conclusion:** Hard assignment on the trained 70-D codes achieved ~70.8%, demonstrating that the shift calibration successfully prevented the representation collapse (28–35%) seen in earlier 70-D attempts. However, both architectures failed to surpass the frozen-feature baseline. F1 refinement hurts the trained codes (dropping them to 66–68%), whereas it adds ~7 points to the frozen features.

---

## Part 2: Execution guide (GPU pipeline). Run only after the Phase 0 verification

### Pre-declared protocol (do not change after seeing results)
1. Teacher: inductive K=8 k-means (16 train patients, 448 input, 28x28 grid), seeds 0, 1, 2. Pseudo-label = the hard cluster map upsampled to the 448 body-crop frame. Liver cluster named by N3 (label-free).
2. Student: ResNet50 U-Net with encoder_weights=None (primary arm). Record the settings of train_gpu_end_to_end.py before running and leave them unchanged. An ImageNet-weights arm may be run only as a separately labelled arm (supervised pretraining).
3. Training data: only the 16 train patients' images and pseudo-labels; no masks and no validation images. Fixed 50 epochs, final epoch only, no validation-based checkpoint selection (a rule, not an omission). Seeds 0, 1, 2.
4. Frame: train at 256x256; map predictions back to the 448 body-crop frame with one fixed, stated interpolation (bilinear on probabilities), identical for all runs.
5. Stage A (no masks opened): predict the 4 validation patients, map to liver with the teacher's N3 naming, apply the F1 post-processing to the student's soft liver probability (also report the hard argmax), and seal the predictions with a manifest hash.
6. Stage B (only after all runs are sealed): liver IoU and Dice per patient, mean +- SD (pooled-pixel per patient, then across patients; also mean-slice), at 224 (primary) and native 512. Compare with F1 inductive 77.96 +- 2.07 and hard K-means 70.83 +- 1.92, using the same metric definition.
7. Criterion: success if the mean over seeds of the F1 pooled-pixel IoU is at least 80.03 and no seed is below 77.96; partial if the mean exceeds 77.96; otherwise failure. No tuning, no extra arms, no selection by validation score; report every run, including failures.
8. Environment: create a new virtual environment for this task, install the packages below only in it, and record the versions.


