# Label usage

**Current execution status:** no training, naming of a new fit, validation prediction generation or evaluation has run. Only no-update sanity tests on a training image and synthetic tensors have executed. Assertions below describe enforced stage boundaries, not nonexistent completed experiment results.

- TRAINING: no liver masks used; access guard rejects annotations and saved label-based scores/mappings. Train patients only.
- CORRESPONDENCE LEARNING: no liver masks used; positives derive from known two-view geometry, negatives from spatial distance.
- DENSE DECODER: no liver masks used; receives frozen tokens (proposed token-only configuration).
- CHECKPOINT SELECTION: no liver masks used; fixed training-image correspondence loss only. No validation images.
- CLUSTER NAMING: no liver masks used; train-image N1/N2/N3 only. No saved semantic reference mapping chooses a cluster.
- INFERENCE: no liver masks used; naming and parameters already fixed, predictions saved before evaluation.
- EVALUATION: liver masks may be used only after all predictions are generated and sealed, for scoring/diagnostics/figures. Evaluate.py verifies the complete six-run manifest before its first mask load.

“K=8 is fixed from the existing experimental protocol and is not dynamically selected using validation IoU in this experiment.”

Historical K/resolution/window/F1 choices were informed by this cohort's labeled evaluations. The pretrained vision-language backbone also had image-text supervision. Do not claim that the entire research history or backbone pretraining was label-independent.

`runtime.py` records read paths, refuses writes outside this experiment, disables network access, blocks annotations/score/reference-mapping files before evaluation, and rejects validation image paths in training/fitting/naming/development. It is a process-level guard, not an independent OS-level security attestation. Existing feature storage contains both splits, but training/fitting/naming code indexes only the train16 records.

`logs/sanity_access.json` and `logs/sanity.json` document the first checks. `integration_checks.py` deliberately attempts a forbidden mask open to test the guard: it is rejected before opening, and no mask values are read. No optimizer step occurs in these tests.
