"""Record bounded experiment results without choosing by evaluation scores."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'audit/scratch/fixes3'
paths=[OUT/'fixes3_A_seed0_retry1_result.json',OUT/'fixes3_B_seed0_result.json',OUT/'fixes3_C_seed0_result.json']
rows=[json.loads(p.read_text()) for p in paths]
assert {r['run'] for r in rows}=={'A','B','C'}
assert all(r['batches']<=500 and r['train_mask_reads']==0 and r['linear_parameters_unchanged'] for r in rows)
assert len({tuple(r['fixed_samples']) for r in rows})==1 and all(r['evaluated_images']==472 for r in rows)
lines=['\n## Task 5 — completed experiments','', 'Producer: `audit/scratch/fixes3_experiment.py`; hard budget500 batches, actual project entrypoint and validation hooks. The first A attempt failed before training; A below is the successful fresh-name retry.','',
'| Run | Batches / optimizer steps | Early stop | Code std | Cross-image cosine | Entropy (nats) | Cluster liver IoU / Dice % | Reused frozen linear IoU % |',
'| --- | --- | --- | --- | --- | --- | --- | --- |']
for r in rows:
 m=r['monitors'];v=r['cluster_metrics']
 lines.append(f"| {r['run']} | {r['batches']} / {r['optimizer_steps']} | {r['early_stopped']} | {m['code_std']:.8f} | {m['cross_image_cosine']:.8f} | {m['cluster_entropy']:.6f} | {v['test/cluster/liver_IoU']:.4f} / {v['test/cluster/liver_Dice']:.4f} | {r['supervised_reference']['liver_IoU']:.4f} |")
for r in rows:
 lines+=['',f"### {r['run']} — {r['name']}",f"- Usage fractions (IDs0–7): {r['monitors']['cluster_usage']}",f"- Evaluation-only majority map: {r['cluster_to_class']}",f"- Overlay: [{Path(r['overlay']).name}]({r['overlay']})",f"- New checkpoint: `{r['checkpoint']}`",f"- Summary: `{OUT/(r['name']+'_result.json')}`",f"- Runtime: {r['elapsed_seconds']:.2f}s; no existing checkpoint/data/cache mutation."]
lines+=['','All runs use final normalized frozen features and restored-mean centering; only correspondence shifts differ. Loss weights, seed, data, batch size, learning-rate schedule and initial model/probe setup are shared. No validation metric controls checkpoint selection: the scratch runner saves only the terminal checkpoint. Labels appear only in evaluation mapping/metrics and GT overlays. The60.8896% linear reference comes from the previously authorized DIAG supervised fit on100 validation slices, and is NOT a per-run supervised probe nor directly the same472-slice evaluation scope. Current logger linear metrics are from an untrained, unchanged head and must not be interpreted as that reference.','',
'Task5 before/after: A empirically reproduces collapse with the current shifts; calibrated B/C are verified through the reported completed batches using actual training, monitor hooks and full validation. This is not evidence of convergence or final quality.','',
'Changed-file anchors: task1 `src/utils.py:272,302`, `src/eval_segmentation.py` metrics_and_mapping output, `src/configs/train_config.yaml` extra_clusters; task2 `src/collapse_monitor.py:6,42`, `src/train_segmentation.py:326,443,460`; task3 `src/validation_visuals.py:8,17`; task5 `src/unlabeled_training.py:6`, `src/train_segmentation.py:281,635`, `src/configs/train_config.yaml:107`.']
with (ROOT/'audit/FIXES3.md').open('a') as f:f.write('\n'.join(lines)+'\n')
print('\n'.join(lines[:9]))
