# Dense STEGO-style correspondence experiment

Status: implementation and pre-training checks; no learned-result claim yet.

The original baseline is unchanged. Every new file lives in this experiment directory.

## Architecture fixed before evaluation

The exact existing local BiomedCLIP checkpoint is loaded through `src/modules.py::BiomedCLIPFeaturizer`. All parameters, including its unused legacy projection, are frozen and excluded from the optimizer. Its final LayerNorm patch output is L2-normalized per location. It stays in eval mode inside no_grad.

Input: B×3×448×448 → frozen ViT-B/16 → B×768×28×28.

Arm A: Conv1×1(768,128), GELU, Conv1×1(128,32); residual block consisting of depthwise Conv3×3, GroupNorm(4), GELU and Conv1×1. L2 output: B×32×28×28. **104,000 trainable parameters**. This arm intentionally retains the baseline spatial bottleneck.

Arm B: same projection/residual block; Conv1×1(32,16); four learned upsampling blocks. Each block is Conv1×1(16,64), PixelShuffle(2), GroupNorm(4), GELU and a depthwise3×3/pointwise1×1 residual refinement. Resolutions: 28 → 56 → 112 → 224 → 448. Final Conv1×1(16,32), channel L2. Output B×32×448×448. Token-only variant: **111,408 trainable parameters**. No resized categorical mask enters the decoder.

An optional CT-image branch is implemented but **disabled in the proposed protocol and not tested or authorized as the final experiment**. If explicitly requested, it changes the dense architecture and must be fixed before training/evaluation; its presence must be reported as an additional high-resolution input, not as information recovered from tokens.

No linear probe, segmentation-label loss, text loss, CRF loss, reconstruction loss, salience sampling or GT-based pixel selection is used.

## Objective

Two independent known affine views of the same windowed/body-cropped CT image are encoded separately by the frozen backbone. Let T_A,T_B map view coordinates into canonical crop coordinates. Sample 96 uniform canonical locations in their shared valid interior; use q_A=T_A^-1(p), q_B=T_B^-1(p). Bilinearly sample and L2-normalize the learned maps at these positions.

Positive pair: the same canonical location in the two views. Negatives: the other sampled canonical locations at Euclidean distance at least0.15 in normalized[-1,1] crop coordinates. Closer locations are excluded, not falsely counted as negatives. Bidirectional InfoNCE averages A→B and B→A cross-entropy, temperature0.10. No cross-patient anatomical matching is asserted. Same-organ locations can be negatives: this is an explicit limitation of instance/location correspondence.

This is **STEGO-inspired geometry correspondence**, not literal Paper A's three correlation-loss terms or restored-mean affinity. No claim of faithful STEGO equivalence is made.

## What density establishes

Arm B produces a learned448×448 feature field before clustering. Native512 assignment uses the existing crop-to-native geometry. It does not resize a28×28 categorical mask. However, subpixel convolution can generate detail from coarse/contextual information and learned phase structure; it cannot guarantee recovery of information discarded by patchification. Report actual raw cluster detail, unchanged-F1 results, geometry loss, mask replay and held-out metrics. Do not equate output size, smoothness or nonconstant sub-patch values with proven anatomical accuracy.
