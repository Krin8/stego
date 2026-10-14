from common import *
import matplotlib.pyplot as plt
m=sealed_manifest();runs={f'{fit}_s{s}':json.loads((OUT/f'{fit}_s{s}_scores.json').read_text()) for fit in ['inductive','transductive'] for s in [0,1,2]}
rows=[]
for key,r in runs.items():
 for rule in ['N1','N2','N3','reference_majority','reference_optimal']:
  row=r['rules'][rule]
  for variant in ['hard','F1']:
   for res in ['224','512']:
    for agg in ['pooled','mean_slice']:
     rows.append(dict(rule=rule,fit=r['fit'],seed=r['seed'],variant=variant,resolution=res,aggregation=agg,selected_cluster=row['selected_cluster'],agreement=row['agreement'],**row['results'][variant][res]['summary'][agg]))
abstention={}
for fit in ['inductive','transductive']:
 for rule in ['N1','N2','N3']:
  rs=[runs[f'{fit}_s{s}'] for s in [0,1,2]];n=sum(r['rules'][rule]['abstained'] for r in rs);slices=sum(r['slices'] for r in rs);empty=sum(r['slices'] for r in rs if r['rules'][rule]['abstained'])
  abstention[f'{fit}/{rule}']=dict(abstained_fits=n,total_fits=3,fit_rate=n/3,empty_scored_slice_instances=empty,total_scored_slice_instances=slices,slice_rate=empty/slices)
dump('summary.json',dict(producer='report.py',manifest_sha256=sha(OUT/'stageA_manifest.json'),rows=rows,abstention=abstention))
def fmt(v):return f"{100*v['mean']:.3f} ± {100*v['std']:.3f} [{100*v['min']:.3f}–{100*v['max']:.3f}]"
def agree(row,res='224'):
 if row['agreement'] is None:return 'Reference'
 ps=row['agreement'][res];return ' / '.join(f"{sum(v[r] for v in ps.values())}/{len(ps)}" for r in ['majority','optimal'])
lines=['# Label-free naming of frozen K8 CT clusters','',
'No gradient training, installs, network, changes to source/datasets/caches/checkpoints. All six requested saved fits exist; no refit was performed. CPU only. Names are fixed once per fit from its own image set, before opening labels.','',
'## Sealed Stage A','',f"Manifest SHA256: `{sha(OUT/'stageA_manifest.json')}`. `stageA_manifest.json` contains every parameter, all selected clusters and hashes of all StageA files. Detached self-hash: `stageA_manifest.sha256`. The only hash exclusions are the manifest itself and its self-hash sidecar. StageB verifies every hash before scoring. `stageA_ledger.json` lists opened paths and blocked attempts. StageB start record: `stageB_started.json`.",
'', '## Protocol details','',
'Frozen BiomedCLIP final normalized patch tokens, L2 and uncentered, body-cropped448 input /28×28, K8 seeds0/1/2. Existing inductive centres fit on train16 (1869 slices), transductive centres fit on all20 (2341 slices). CT[-150,250] HU preprocessing is unchanged. StageA reads only image inventory/association metadata, images, raw DICOM, features, centres, label-free fit metadata, model/tokenizer files and code. It does not read the saved semantic mappings. Metadata includes unused mask filenames/hash fields, not mask values; those mask files are never opened.',
'', 'N1:200 body patches nearest each centre (across all body patches, not restricted by hard cluster assignment) (squared Euclidean, ties image index then patch index); body membership uses patch centre in N2 body mask.112×112 native-windowed crops, nearest-integer centre, zero padding,224 bilinear antialias,3 channels, BiomedCLIP mean/std once. Full CLIP image projection gives normalized512D crop vectors, averaged and renormalized for cosine. Text prompts exactly as specified. Population z-score each prompt across8 clusters, zero variance→0. Argmax liver z-score names one cluster. Top prompt, z-margin and raw-cosine margin per cluster are saved in *_naming.json. Because liver z-scoring is a positive affine transform across clusters, the liver argmax also equals the raw liver-cosine argmax; z-scoring matters for cross-prompt ranking.',
'', 'N2:raw nativeHU>-500 → opening disk5 → largest8-connected component → fill holes (4-connected background). Normalize patch-centre x/y to body bounding box. Eligible patches have centre inside body. Each patch counts once for area fraction/centroid. MeanHU is the arithmetic mean of patch-body-footprint HU means, using unwindowed HU. Candidate x<.5, y<.6, HU40..180 inclusive; choose largest area, tie lowestID; none→abstain. N3 accepts only exact N1/N2 agreement, else abstains. These operational conventions were fixed before scoring; full values are in PARAMETERS.json and manifest.',
'', 'StageB: hard assignment and original F1, changing only the frozen selected semantic subset. F1 uses the original saved per-fit temperature and identical soft weights, inverse crop, guided radius8/eps.01,>=.5,LCC8,hole fill4,disk3 closing. Abstentions are empty masks, including F1. No tuning follows labels.',
'', 'Reference: inductive uses stored train16 strict-majority mapping and train-confusion optimal subset (both resolutions same native-derived mapping, as FINAL_PIPELINE); transductive uses saved other19 LOPO majority/optimal mappings independently at224/512. FINAL_BASELINE was the earlier224-input baseline; the relevant saved448 reference is IMPROVE/FINAL_PIPELINE. Agreement is exact singleton/subset equality, per heldout patient; abstention agrees only if the assisted subset is empty.','',
'## Main table: primary224 pooled-pixel scores (%)','',
'Each patient pools TP/FP/FN across slices; report equal-patient mean/sample SD/min/max, separately for each fit and seed. Inductive scores4 validation patients; transductive scores20. Agreement columns count patients whose named subset exactly equals the assisted subset (majority / optimal). The full report also gives mean-slice metrics and native512 scores.','',
'| Rule | Fit | Seed | Selected cluster | Assisted agreement M/O | Hard IoU | Hard Dice | F1 IoU | F1 Dice |','|---|---|---:|---|---|---:|---:|---:|---:|']
for key,r in runs.items():
 for rule in ['N1','N2','N3','reference_majority','reference_optimal']:
  row=r['rules'][rule];h=row['results']['hard']['224']['summary']['pooled'];f=row['results']['F1']['224']['summary']['pooled'];selected=row['selected_cluster'] if row['selected_cluster'] is not None else 'abstain'
  lines.append(f"| {rule} | {r['fit']} | {r['seed']} | {selected} | {agree(row)} | {fmt(h['iou'])} | {fmt(h['dice'])} | {fmt(f['iou'])} | {fmt(f['dice'])} |")
lines+=['','## Abstention','', '| Fit / rule | Abstained fits | Fit rate | Empty-scored slice instances | Slice rate |','|---|---:|---:|---:|---:|']
for key,v in abstention.items():lines.append(f"| {key} | {v['abstained_fits']}/{v['total_fits']} | {100*v['fit_rate']:.3f}% | {v['empty_scored_slice_instances']}/{v['total_scored_slice_instances']} | {100*v['slice_rate']:.3f}% |")
lines+=['','## All resolutions / aggregations (%)','', '| Rule | Fit | Seed | Prediction | Scale | Aggregation | IoU mean ± SD [min–max] | Dice mean ± SD [min–max] |','|---|---|---:|---|---:|---|---:|---:|']
for r in rows:lines.append(f"| {r['rule']} | {r['fit']} | {r['seed']} | {r['variant']} | {r['resolution']} | {r['aggregation']} | {fmt(r['iou'])} | {fmt(r['dice'])} |")
lines+=['','## Per-patient scores (%)','', '| Rule | Fit | Seed | Patient | Prediction | Scale | Pooled IoU/Dice | Mean-slice IoU/Dice |','|---|---|---:|---|---|---:|---:|---:|']
for key,r in runs.items():
 for rule,row in r['rules'].items():
  for variant in ['hard','F1']:
   for res in ['224','512']:
    for pat,v in row['results'][variant][res]['patients'].items():
     lines.append(f"| {rule} | {r['fit']} | {r['seed']} | {pat} | {variant} | {res} | {100*v['pooled']['iou']:.3f}/{100*v['pooled']['dice']:.3f} | {100*v['mean_slice']['iou']:.3f}/{100*v['mean_slice']['dice']:.3f} |")
lines+=['','## Cluster ranking — label-using diagnostic only','', 'Ranked by liver precision, with recall against all evaluated liver pixels. No ranking value is fed back to names. Inductive uses4 validation patients; transductive uses20. Both resolutions retained in each *_scores.json. Native512 table follows.','', '| Fit | Seed | Rank | Cluster | Liver precision % | Liver recall % | Pixel usage % |','|---|---:|---:|---:|---:|---:|---:|']
for key,r in runs.items():
 for rank,v in enumerate(r['cluster_ranking']['512']['order_by_precision'],1):lines.append(f"| {r['fit']} | {r['seed']} | {rank} | {v['cluster']} | {100*v['precision']:.3f} | {100*v['recall']:.3f} | {100*v['pixel_fraction']:.3f} |")
rr=json.loads((OUT/'image_records.json').read_text());fixed=json.loads((OLD/'figure_selection.json').read_text())['eight_indices'];truth=np.load(OLD/'scoring_masks_native.npy',mmap_mode='r');figpaths=[]
for rule in ['N1','N2','N3']:
 fig,ax=plt.subplots(8,8,figsize=(28,28))
 for row,i in enumerate(fixed):
  im=raw_image(rr[i]);ax[row,0].imshow(im,cmap='gray',vmin=0,vmax=1);ax[row,0].set_title(rr[i]['stem']);ax[row,1].imshow(im,cmap='gray',vmin=0,vmax=1);ax[row,1].contour(truth[i],levels=[.5],colors=['red'],linewidths=.6);ax[row,1].set_title('Ground truth')
  for col,(key,r) in enumerate(runs.items(),2):
   with np.load(OUT/f'{key}_figure_masks.npz') as panel:pred=panel[f'{rule}_hard_{i}'].astype(bool)
   ax[row,col].imshow(im,cmap='gray',vmin=0,vmax=1)
   if pred.any():ax[row,col].contour(pred,levels=[.5],colors=['lime'],linewidths=.6)
   metric=scores(confusion(pred,truth[i],2),[1]);cl=r['rules'][rule]['selected_cluster'];ax[row,col].set_title(f'{key} C={cl}\nIoU={100*metric["iou"]:.1f}%')
  for a in ax[row]:a.axis('off')
 fig.suptitle(rule+' — frozen named hard cluster, green prediction; GT red',fontsize=18);fig.tight_layout();path=OUT/f'{rule}_eight_slices.png';fig.savefig(path,dpi=120);plt.close(fig);figpaths.append(str(path))
lines+=['','## Figures','', 'Same8 fixed validation slices, one figure per rule; columns input, GT and all6 fit/seed outputs. Green hard-prediction contour, per-slice nativeIoU; blank contour denotes abstention/empty prediction. Slice selection was inherited and historically used labels; it does not choose naming parameters.']
for rule,path in zip(['N1','N2','N3'],figpaths):lines+=['',f'![{rule}]({Path(path).name})']
ledger=dict(StageA=dict(label_mask_reads=0,prior_mapping_score_reads=0,inputs='raw CT, windowed images, image association metadata, frozen features, centres, label-free fit metadata, local backbone/tokenizer',decisions='N1/N2/N3 selections frozen once per fit own images'),StageB=dict(label_use='scoring, cluster precision/recall ranking, agreement against pre-existing assisted mappings and figure overlays only',mapping_reference='inductive train16 stored labels/confusion; transductive other19 saved LOPO mappings',parameter_changes_after_start=False),manifest_sha256=sha(OUT/'stageA_manifest.json'),producer='stage_a.py / seal.py / stage_b.py / report.py',reruns='No scoring-driven rerun; pre-scoring execution repairs, if any, recorded in execution notes')
dump('label_use_ledger.json',ledger);dump('figures.json',dict(producer='report.py',indices=fixed,paths=figpaths))
lines+=['','## Label-use ledger','',json.dumps(ledger,indent=2),
'', '## UNVERIFIED / limitations','',
'These patients informed earlier choices. K8,448 resolution and HU window were label-selected historically, so this is not an untouched test. Label-free naming now does not erase those choices. New-cohort/scanner generalization, absence-of-liver behavior and clinical use remain UNVERIFIED. Native image-left anatomical prior assumes the documented axial orientation; no patient-orientation canonicalization is performed. Patch-centre/body-boundary discretization and top200/crop112 rules are fixed assumptions, not tuned alternatives. N1 z-scores and margins are descriptive, not calibrated probabilities. No gradients or head training were run.']
(OUT/'NAMING.md').write_text('\n'.join(lines)+'\n');print('REPORT DONE',elapsed())
