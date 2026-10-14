# STEGO fidelity and deliberate deviations

Reference: Hamilton et al., arXiv:2203.08414v1, equations 1–5, sections 3.1–3.3 and A.10. Official implementation pinned to `eb4d6b521740bd4265681b353547f2ffca65d673`; selected pipeline files and MIT license are preserved under `reference/`. `reference/manifest.json` records their digests. Inspected before implementation.

## Actual mechanism

Frozen visual features have shape B×768×28×28. The head adds Conv1×1(768,70) and Conv1×1(768,768)→ReLU→Conv1×1(768,70), yielding B×70×28×28. Exactly 698,252 parameters are trainable. Each projection branch has an independent Dropout2d(.1); another independent dropout supplies the correspondence signal. This matches reference modules.py:73–81,108–118.

`reference_loss.py` copies the exact functions/class from reference modules.py:275–398, omitting only the TorchScript decorator on `super_perm`. It draws two independent 11×11 coordinate samples, uses bilinear grid_sample with border padding and align_corners=True, normalizes sampled features/codes and computes all pairwise cosine correlations. The reference helper uses restored-global-mean spatial centering: subtract each source point's target mean, then restore the original global mean. This is the pinned-reference variant, not literal paper Eq(3).

Each component averages `−(centered feature cosine − shift) × max(code cosine,0)`. Self uses the same image/sample twice; positive-inter uses a uniformly chosen top-seven image neighbor; negative-inter uses the exact reference self-excluding `super_perm` in the current batch, five times. That helper can repeat target indices after repairing self-pairs; this is preserved. There is no InfoNCE or temperature in the training loss.

Active pinned-reference defaults are fixed, without looking at new validation results: weights self=.67, neighbor=.25, random=.63; shifts self=.18, neighbor=.12, random=.46; correspondence weight=1. Adam lr=.0005, default betas/epsilon, no weight decay or scheduler. Batch16 matches pinned config; paper A.10 says32. The paper's dataset-specific Table6 settings differ from this repository's active defaults. Previously calibrated C/D/E shifts are NOT used in this new fidelity experiment. Successful transfer or avoidance of collapse is not assumed.

## Two views: explicit resolution of the request

User confirmed the original three losses with alignment OFF. The core reference does not designate hard geometric positive/negative patch pairs; attraction is determined by feature affinity. The optional negative cosine alignment term is isolated in `geometry.py` and never added by `train.py`. Its coordinate-grid transpose compensates for reference sample()'s spatial transpose; identity/flip tests verify correspondence. No generic geometric InfoNCE or extra decoder is substituted. There are no augmented-image forward passes in the primary experiment, consistent with inactive reference augmentation alignment.

## Necessary and user-requested deviations

- DINO replaced by exact local frozen BiomedCLIP checkpoint; final LayerNorm patch tokens, CLS excluded. Position embeddings interpolate at448; resulting grid28. Preprocessing and patch L2 normalization match the existing B2-448 baseline. Reference feeds unnormalized last-layer tokens to its head and pools those for image neighbors; here both head and neighbor pooling use baseline-normalized tokens. This is an explicit controlled-baseline adaptation, not exact DINO feature scaling.
- CHAOS CT native HU window[-150,250], existing body square crop and5% margin, grayscale replication, BiomedCLIP mean/std once; no color jitter, five-crop expansion or DINO preprocessing. Train/eval448 instead of original224/320. Native mask outputs preserve inverse-crop geometry.
- Frozen deterministic tokens are read from the existing verified B2-448 array. Training indexes only the1869 train rows; validation rows are not sampled. Source cache is never written. One real-image replay checks exact numerical equality; all-row re-extraction is not claimed.
- Image kNN is computed in a new experiment-only array over train images, L2(GAP(tokens)) cosine, top30 stored and top7 sampled, explicit self exclusion. No existing cache is modified. The reference includes self in top30 and the data loader skips it by rank.
- Budget is user-confirmed5000 **head optimizer batches** per seed0/1/2. Pure PyTorch replaces Lightning. Original multiple-optimizer global_step units are not reproduced. There is no extra linear or cluster-probe optimizer.
- Checkpoint: minimum fixed train-only correspondence loss, dropout off, one fixed image per train patient and fixed neighbors, RNG reset at every50 batches; ties keep earliest. Validation images/labels never enter this criterion. Collapse flags (cosine>.99 or normalized pixel std<.01 for3 consecutive checks) are reported, not used to retune or replace seeds; all5000 batches remain the fixed budget.
- Posthoc Euclidean KMeans on L2-normalized70-D codes, K8, n_init3,max_iter100,tol1e-4,Lloyd, all train patches. This deliberately replaces the original online cosine probe, as requested for the controlled baseline comparison.
- N1/N2/N3 naming and fixed F1 replace original Hungarian class matching and CRF. Naming uses only train images, native HU and frozen text/image embeddings; no masks. F1 soft assignment temperature is the inherited train-distance gap/log4 statistic, not a correspondence-loss temperature.
- No high-resolution learned decoder. Raw learned representation and cluster grid remain28×28. F1 resampling/refinement does not establish learned subpatch precision or guarantee less blocky/better anatomical masks.

## Validation gates

`checks.py`: exact pinned-loss and gradient parity; head dropout RNG parity; frozen backbone deterministic replay; zero backbone gradients; head gradients finite; label access guard.
`integration_checks.py`: bit-exact inherited F1 on synthetic70-D inputs, outside-crop geometry, empty abstentions, disjoint16/4 patients.
`preflight.py`: hashes config, source, exact backbone, existing feature array and baseline seal before training. `seal_predictions.py` hashes all3 seeds' predictions/names before a separate evaluation process can open masks.
