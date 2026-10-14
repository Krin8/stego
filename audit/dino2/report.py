from common import *
weights=json.loads((OUT/'weight_verification.json').read_text());sel=json.loads((OUT/'pair_selectivity.json').read_text()) if (OUT/'pair_selectivity.json').exists() else {'results':{}}
rows=[];per_seed=[]
for arm in ['D2','D3']:
 if not (OUT/f'{arm}_extract.json').exists():continue
 for k in [8,2]:
  runs=[json.loads((OUT/f'{arm}_k{k}_s{s}_scores.json').read_text()) for s in [0,1,2]]
  for res in ['224','512']:
   pats=list(runs[0]['results'][res]['folds'])
   for mapping in (['majority','optimal'] if k==8 else ['Hungarian']):
    patient={pat:{m:float(np.mean([r['results'][res]['folds'][pat][mapping][m] for r in runs])) for m in ['iou','dice']} for pat in pats}
    rows.append(dict(arm=arm,grid='28×28',K=k,mapping=mapping,resolution=res,patients=patient,**{m:patient_stats([v[m] for v in patient.values()]) for m in ['iou','dice']},auc=sel['results'][arm]['auc']))
    for s,r in enumerate(runs):per_seed.append(dict(arm=arm,K=k,seed=s,mapping=mapping,resolution=res,**{m:patient_stats([r['results'][res]['folds'][pat][mapping][m] for pat in pats]) for m in ['iou','dice']}))
previous=json.loads((ROOT/'audit/dino/summary.json').read_text())['rows']
for r in previous:
 if r['arm'] in ['D1','B0 (historical)','B2_448 (historical)']:
  row=dict(r);row['arm']=r['arm']+' reference'
  if r['arm']=='B2_448 (historical)' and sel['results']:row['auc']=sel['results']['BiomedCLIP_B2_448']['auc']
  rows.append(row)
qualifying=[r for r in rows if r['arm'] in ['D2','D3'] and r['K']==8 and r['resolution']=='224' and r['iou']['mean']>=.6287 and r['auc']>=.9]
beats=[r for r in rows if r['arm'] in ['D2','D3'] and r['K']==8 and r['resolution']=='224' and r['iou']['mean']>.6978]
rule=dict(justified=bool(qualifying),qualifying=[dict(arm=r['arm'],mapping=r['mapping'],iou=r['iou']['mean'],auc=r['auc']) for r in qualifying],beats_BiomedCLIP_threshold=[dict(arm=r['arm'],mapping=r['mapping'],iou=r['iou']['mean']) for r in beats],qualification='Threshold crossing only; inspect K2 and require label-free naming test before any claim',head_training_run=False)
dump('summary.json',dict(producer='report.py',rows=rows,per_seed=per_seed,decision=rule,pair_selectivity=sel))
def fmt(v):
 digits=6 if v['mean']>0 and v['mean']<1e-5 else 3
 return f"{100*v['mean']:.{digits}f} ± {100*v['std']:.{digits}f} [{100*v['min']:.{digits}f}–{100*v['max']:.{digits}f}]"
lines=['# D2 / D3 frozen-feature CT liver results','',
'Parameters were locked in `PROTOCOL.json` before scoring. Producers: `verify_weights.py`, `extract.py`, `figures.py`, `fit_score.py`, `pair_affinities.py`, `pair_score.py`, `report.py`. Executed commands are in `commands.txt`. No gradients, network, downloads, installs, or modifications to project source / datasets / existing caches / weight files.',
'', '## Weight verification','', '| Arm/file | Supplied hub commit | Expected SHA256 | On-disk SHA256 | Match |','|---|---|---|---|---|']
for arm,r in weights['arms'].items():
 lines.append(f"| {arm} model.safetensors | {r['hub_commit_user_supplied']} | {r['expected_sha256']} | {r['actual_sha256']} | {r['digest_matches']} |")
 if arm=='D2':
  b=r['source_bin'];lines.append(f"| D2 original pytorch_model.bin (hash only, not loaded) | {r['hub_commit_user_supplied']} | {b['expected_sha256']} | {b['actual_sha256']} | {b['matches']} |")
lines+=['','Source: `weight_verification.json`. Both digests checked before any model load. Hub commits are the user-supplied provenance, not independently verified over the network. D2 converted safetensors has198 tensors; the original binary was hashed, never deserialized.','', '## Preprocessing and feature tap','',
'Identical geometry to D1: native512 CT derivatives already windowed HU[-150,250] → [0,1], round(65535*x) uint16 storage then /65535; no second window. Image-only body mask >0.05, largest8-connected component, hole filling, square crop with5% margin on each side, zero padding. Bilinear antialiased resize to448 (D2) or392 (D3), grayscale replicated3 channels. ImageNet mean[0.485,0.456,0.406], std[0.229,0.224,0.225] read from each local processor JSON and applied exactly once. The processors\' stock224/256 resize/crop and /255 settings are NOT applied: they would violate the declared geometry or double-rescale already normalized pixels.',
'', 'D2: local transformers.ViTModel, add_pooling_layer=False, interpolate_pos_encoding=True. D3: local transformers.Dinov2Model, internal position interpolation. Both return785 last_hidden_state tokens including CLS, after final LayerNorm (verified with a forward hook). Drop CLS, L2 normalize784 patch tokens, uncentered, reshape28×28×768. D1 used the same semantic tap: project get_intermediate_layers(n=1) applied final LayerNorm before CLS removal and L2.',
'', '## Every difference from D1','']
protocol=json.loads((OUT/'PROTOCOL.json').read_text())
lines += ['- '+s for s in protocol['differences_from_D1']]
lines += ['','D1 used the project eager-attention implementation and LayerNorm epsilon1e-6; D2 uses transformers SDPA with supplied config epsilon1e-12, and D3 uses transformers SDPA with supplied epsilon1e-6. These inherited differences were not tuned. All other fitting, mapping, mask inversion, scoring, pair sampling/centering and PCA settings are retained. `reuse_provenance.json` records hashes of reused D1 scripts; `fit_score.py` and `figures.py` differ only by arm arguments and output names. `common.py` additionally protects the pre-existing weights directory and serializes empty loader-diagnostic sets. The D2 log-writing retry did not alter any model setting.','', '## Load checks and timing','', '| Arm | Missing / unexpected / mismatched keys | Frozen / eval | Exact repeat | Token shape | Batch8 forward seconds | Full projection minutes | Actual extraction minutes | Backend / LayerNorm eps |','|---|---|---|---|---|---:|---:|---:|---|']
for arm in ['D2','D3']:
 if (OUT/f'{arm}_checks.json').exists():
  c=json.loads((OUT/f'{arm}_checks.json').read_text());e=json.loads((OUT/f'{arm}_extract.json').read_text()) if (OUT/f'{arm}_extract.json').exists() else {}
  info=c['strict_loading_info'];lines.append(f"| {arm} | {info.get('missing_keys',[])} / {info.get('unexpected_keys',[])} / {info.get('mismatched_keys',[])} | {c['all_parameters_frozen']} / {c['model_eval']} | {c['deterministic_exact']} (max error {c['max_abs_difference']}) | {c['including_CLS_shape']} | {c['batch8_forward_seconds']:.4f} | {c['projected_full_seconds']/60:.3f} | {e.get('forward_extraction_seconds',0)/60:.3f} | {c['attention_backend']} / {c['layer_norm_eps']} |")
 if (OUT/f'{arm}_FAILED.json').exists():lines+=['',arm+' stopped: '+(OUT/f'{arm}_FAILED.json').read_text()]
lines+=['','Timing includes a warmed batch8 forward; full projection includes measured preprocessing time, not model loading or k-means/scoring. Neither completed arm exceeded90 minutes, so no200-slice fallback was necessary. Actual parameters are recorded before scoring in each *_checks.json and *_parameters.json. Installed model-forward and interpolation source is saved in *_model_forward_source.txt.','', '## Segmentation (%)','',
'All2341 slices/20 patients fit label-free KMeans. K8/K2, seeds0/1/2, k-means++, n_init3, max_iter100, tol1e-4, Lloyd, float32; no post-processing. Leave one patient out of mapping: other19 labels determine strict-majority (>50%, ties background), exhaustive256-subset IoU-optimal K8 mapping, or one-to-one K2 Hungarian maximizing pixel agreement. Heldout labels only score. Outside-crop pixels are forced background. Native nearest inversion and224 nearest resize; mappings fitted independently at each scoring resolution, exactly as D1/IMPROVE.',
'', 'IoU/Dice pool pixels within patient. Average seeds within each patient; then equal-patient mean, sample SD, min/max over20 patients. Complete per-patient mappings/confusions: *_scores.json; every seed also listed below.','', '| Arm | Grid | K | Mapping | Scale | IoU mean ± SD [min–max] | Dice mean ± SD [min–max] | AUC |','|---|---|---:|---|---:|---:|---:|---:|']
for r in rows:lines.append(f"| {r['arm']} | {r['grid']} | {r['K']} | {r['mapping']} | {r['resolution']} | {fmt(r['iou'])} | {fmt(r['dice'])} | {r['auc']:.6f} |")
lines+=['','References copied from `audit/dino/summary.json`, which aggregates IMPROVE/D1 outputs. B0 AUC uses historical14-grid/full-frame geometry; it is not the matched body-crop28-grid AUC. Matched BiomedCLIP B2-448 AUC is recomputed from unchanged cached features and checked against D1-run evidence.','', '## Same-image pair selectivity','',
'Only472 validation slices/4 patients; labels diagnose pair classes and never select features, sampling, shifts or clustering. Same actual project loss.forward/helper restored-global-mean centering and PAIRS sampler as D1: batch16/last8,7 passes,11×11 coordinates,2048 fixed pair positions. Common28×28 majority labels from body-cropped nearest448 masks, ties background. Overall5/10/20% attraction thresholds fixed by label-free score quantiles before labels are read. AUC/AP and target precision/recall concern liver anchors. Includes same-coordinate/self pairs as D1 did.','', '| Arm | AUC | AP | Overall attraction | Threshold | Liver-target precision % | Liver-pair recall % |','|---|---:|---:|---:|---:|---:|---:|']
for arm,r in sel['results'].items():
 for s in r['sweep']:lines.append(f"| {arm} | {r['auc']:.6f} | {r['average_precision']:.6f} | {100*s['attraction']:.3f}% | {s['threshold']:.6f} | {100*s['liver_target_precision']:.3f} | {100*s['liver_pair_recall']:.3f} |")
lines+=['','## Every seed (%)','','| Arm | K | Seed | Mapping | Scale | IoU mean ± SD [min–max] | Dice mean ± SD [min–max] |','|---|---:|---:|---|---:|---:|---:|']
for r in per_seed:lines.append(f"| {r['arm']} | {r['K']} | {r['seed']} | {r['mapping']} | {r['resolution']} | {fmt(r['iou'])} | {fmt(r['dice'])} |")
lines+=['','## PCA figures','', 'Eight fixed slices unchanged from D1. PCA fit label-free jointly on their features; full SVD,3 components, joint1st/99th percentile color scaling. Ground truth is displayed only; inherited largest-liver slice selection used labels historically. Figures were generated before segmentation scoring.']
for arm in ['D2','D3']:
 if (OUT/f'{arm}_figures.json').exists():lines+=['',f'![{arm} eight fixed PCA slices]({arm}_eight_slices_PCA.png)']
lines+=['','## Declared decision rule','',
f"Justified: **{rule['justified']}**. Requires K8 primary224 mean LOPO liver IoU≥62.87% AND AUC≥0.90. Qualifying arm/mapping rows: {rule['qualifying']}.",
'',f"'Beats BiomedCLIP' threshold (>69.78%) rows: {rule['beats_BiomedCLIP_threshold']}. This phrase denotes the declared numeric flag only. K2 is evaluated above; any qualifying arm still requires the label-free naming test before a claim. No head training was run.",
'', '## UNVERIFIED / limitations','',
'These20 patients informed earlier choices. This is not an untouched test. KMeans is label-free, but cluster naming uses other-patient labels; optimal-subset LOPO is label-assisted mapping. Label-free naming, trained-head benefit, independent-cohort transfer, clinical utility and liver-absent behavior remain UNVERIFIED. Four-patient same-image pair selectivity includes self pairs and does not establish cross-image or cross-cohort selectivity. Hub commits are user-supplied provenance; the on-disk digests are independently verified without network. No unrequested settings, seeds or arms were added.']
(OUT/'DINO2.md').write_text('\n'.join(lines)+'\n');print(json.dumps(rule,indent=2))
