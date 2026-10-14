# Label usage

BiomedCLIP is the existing image-text pretrained checkpoint. This experiment supplies no CHAOS masks to the backbone; it remains frozen. The historical pretraining corpus's membership/overlap with CHAOS is UNVERIFIED, so an absolute claim about what its authors trained on is not made.

STEGO training: train images/frozen features only. No supervised segmentation or linear probe, mask loss, salience signal, text loss, pseudo-label objective or IoU/Dice calculation. Head receives only self, neighbor and random correspondence gradients.

Checkpoint selection: fixed train-only correspondence objective; no validation images or liver labels. K8 and seeds0/1/2 are fixed before runs. All runs are reported; no best seed selected.

Cluster naming: existing N1/N2/N3 on training images only. Text similarities and fixed body/HU/location rules do not read organ masks. Inference and fixed F1 do not read labels. An abstention is an empty prediction, not replaced by a label-selected cluster.

Evaluation: all seeds' predictions and naming artifacts must be complete and SHA256-sealed before `evaluate.py` opens cached ground-truth masks and saved baseline scores. Labels enter scoring and displayed GT only. Historical fixed-eight slice selection included labels; no new slice selection occurs.

The runtime blocks label paths/reference score reads before evaluation, network connections, writes outside the new experiment and validation image reads during development/training/fitting/naming. Each stage logs accessed paths. A synthetic guard test records a blocked attempt, not a successful label read. Inspect the ledger before claiming run completion.

Historical project development used labels to inform K, resolution, window, F1 and other investigations. This is a controlled transfer experiment on previously studied patients, not an untouched test cohort.
