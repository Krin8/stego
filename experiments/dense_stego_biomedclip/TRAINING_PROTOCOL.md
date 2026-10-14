# Training protocol — pending launch

The proposed budget is500 batches per arm/seed, batch2, CPU; seeds0,1,2. `configs/proposed.json` is not a locked authorization to launch. The user was asked to confirm the budget, token-only versus CT-assisted decoder, and the suspended process situation. No training has started.

After those decisions, copy the approved configuration to `configs/locked.json`, set status LOCKED and save its SHA256 in `configs/locked.sha256`. Do not create these files until the outstanding choices are resolved. All six runs use the same fixed protocol. No continuation or extra candidate based on evaluation scores.

Optimizer: Adam, learning rate0.0005, weight decay0; only new model parameters. Cosine schedule T_max equals the fixed batch budget. This is a new experiment's schedule, not a claim to reproduce the old 5,000-step STEGO schedule.

Two-view augmentation: independent isotropic crop scale0.8–1, rotation−5°..5°, translation−0.08..0.08 normalized crop units; bilinear image resampling, border padding. No flip. Gain0.95–1.05 and additive offset−0.02..0.02, then clamp[0,1]. Model normalization once afterward. Positive coordinates exclude view edges beyond±0.9. Augmentations are generated from a seeded torch generator. Native CT window/body crop are unchanged.

Train16 only. Cycle patients evenly; choose each patient's image with the run's fixed RNG. Validation images are excluded from training, monitor loss, clustering, naming and checkpoint selection. Checkpoint criterion: minimum bidirectional correspondence loss on fixed augmented TRAIN image pairs, one image from each of16 train patients, at every50 batches and the last batch. The monitor generator and pairs are fixed. Earliest wins exact ties. Save best.pt and last.pt. No Dice/IoU is available. A train-monitor criterion may favor overfitting and does not certify organ quality.

K=8. K-means fits512 uniform canonical locations from every training image, with spatial sampling fixed across arms/seeds, Lloyd, n_init3, max_iter100, tol1e-4, clustering seed equal to run seed. This bounded spatial sample is shared across the two new arms but differs from the old all-patch baseline; disclose it as a tractability compromise. No masks determine sample locations.

Naming scans ALL eligible body locations from the train images. N1 uses200 nearest candidates per centre, native112 crops,224 joint image embedding and the existing fixed15 text embeddings. N2 retains its−500HU/disk5/body/centroid/HU40–180/area rules. Grid index-to-native transforms use the actual28 or448 grid. If a subpixel cell contains no native pixel, HU is taken from the same rounded native centre used to decide body membership; this mechanical dense-grid adaptation is fixed before labels and logged. N3 abstains on disagreement. No reference semantic mapping is used.

Temperature for F1 is the inherited median nearest/second-nearest centre squared-distance gap divided by log4 (floor1e-6), estimated on the fixed fit sample. The original guided radius8/epsilon.01, threshold.5, largest8-connected2D component,4-connected hole fill and disk3 closing remain unchanged. Only feature-grid reshape is generalized to actual output size.

Inference saves float32 learned feature maps for all472 val slices, raw grid and native cluster assignments, and native hard/F1 masks for every rule. Five fixed images per run are replayed exactly before sealing. After all six runs finish, seal all predictions, naming, centres, chosen checkpoints, config and code. Only then may evaluate.py read masks. Re-run from saved features/checkpoint if exact replay fails; do not tune after labels.

Original baseline artifacts are never overwritten. No package installation/download is needed. Predicted B feature storage is approximately12.1GB per seed; totalB roughly36.4GB. New files only under this experiment.

Pipeline commands (not executed as training):

```sh
PYTHONDONTWRITEBYTECODE=1 python3 experiments/dense_stego_biomedclip/run_suite.py
```

The controller refuses a missing locked configuration, refuses overlapping/suspended training processes, runs sequentially and stops on the first failure. It never kills, restarts or changes settings automatically.
