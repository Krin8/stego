# Dense correspondence experiment — not yet run

No experimental segmentation results exist. Do not interpret the baseline's saved metrics as results for these new models.

Implemented: two frozen-backbone correspondence arms; deterministic geometric positives and masked spatial negatives; new-module-only optimizers; train-only checkpoint criterion; train-only clustering/naming; native predictions with unchanged F1 operations; prediction sealing; separate evaluation; actual-mask figures; baseline comparison and seed-stability reporting.

Verified before training:
- Backbone trainable parameters:0. Backbone gradients:none.
- ArmA new trainable parameters:104,000; output1×32×28×28.
- ArmB token-only new trainable parameters:111,408; output1×32×448×448 through56/112/224/448 stages.
- Optimizer membership is exactly the new module parameters.
- Repeated forward calls are deterministic.
- A training image's frozen features match the saved baseline within the recorded tolerance.
- Known geometric correspondences invert correctly; synthetic aligned pairs have lower loss than misaligned pairs.
- Optimizer steps executed:0. No masks read.

Evidence: `logs/sanity.json`, `logs/sanity.log`, `logs/integration_checks.json`.

BLOCKED before training: budget/decoder input confirmation and suspended STEGO PID52596 (plus suspended caffeinate PID67971). Neither has been terminated. No new training process has been started.

NOT VERIFIED: end-to-end trained checkpoint behavior, k-means/naming on learned features, checkpoint replay after real training, anatomical boundary improvement, actual validation scores, trained seed stability and tiny-liver behavior. The pending experiment must establish these; output resolution alone cannot.
