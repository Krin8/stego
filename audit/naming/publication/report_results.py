"""Publication tables from exact re-evaluation; no method/seed selection or fitting."""
import os,sys,csv,itertools
from pathlib import Path
os.environ['NAMING_STAGE']='B';sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
import matplotlib.pyplot as plt
P=OUT/'publication';manifest=sealed_manifest();audit=json.loads((P/'audit_verification.json').read_text());stability=json.loads((P/'stability.json').read_text());repro=json.loads((P/'reproduction_verification.json').read_text())
runs={(f,s):json.loads((P/f'reproduction/{f}_s{s}_scores.json').read_text()) for f in ['inductive','transductive'] for s in range(3)}
def write_json(name,obj):(P/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def stat(x):return patient_stats(x)
def fmt(x):return f"{100*x['mean']:.3f} ± {100*x['std']:.3f} [{100*x['min']:.3f}–{100*x['max']:.3f}]"
def pd(x):return f"{100*x['iou']:.3f} / {100*x['dice']:.3f}"
def csvwrite(name,rows):
 with (P/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',*['| '+' | '.join(map(str,r))+' |' for r in rows]])
allrows=[];perrows=[];avgrows=[];seedvariation=[];failures=[]
for (fit,seed),run in runs.items():
 for rule,r in run['rules'].items():
  for variant in ['hard','F1']:
   for res in ['224','512']:
    q=r['results'][variant][res]
    for agg in ['pooled','mean_slice']:
     row=dict(fit=fit,seed=seed,rule=rule,selected_cluster=r['selected_cluster'],variant=variant,resolution=int(res),aggregation=agg)
     for metric in ['iou','dice']:
      for k,v in q['summary'][agg][metric].items():row[metric+'_'+k]=100*v
     allrows.append(row)
     for pat,pr in q['patients'].items():perrows.append(dict(fit=fit,seed=seed,rule=rule,variant=variant,resolution=int(res),aggregation=agg,patient=pat,slices=pr['slice_count'],iou=100*pr[agg]['iou'],dice=100*pr[agg]['dice']))
 for variant in ['hard','F1']:
  ss=[s for p in run['rules']['N1']['results'][variant]['512']['patients'].values() for s in p['slice_scores']];empty=[s for s in ss if s['tp']+s['fn']==0];tiny=[s for s in ss if 0<s['tp']+s['fn']<2000];large=[s for s in ss if s['tp']+s['fn']>=2000]
  failures.append(dict(fit=fit,seed=seed,variant=variant,empty_slices=len(empty),empty_any_fp=sum(s['fp']>0 for s in empty),empty_mean_fp=float(np.mean([s['fp'] for s in empty])) if empty else None,tiny_slices=len(tiny),tiny_mean_slice_iou=100*float(np.mean([s['iou'] for s in tiny])),large_slices=len(large),large_mean_slice_iou=100*float(np.mean([s['iou'] for s in large]))))
for fit in ['inductive','transductive']:
 for rule in ['N1','N2','N3','reference_majority','reference_optimal']:
  for variant in ['hard','F1']:
   for res in ['224','512']:
    for agg in ['pooled','mean_slice']:
     ps=[runs[(fit,s)]['rules'][rule]['results'][variant][res]['patients'] for s in range(3)];patients={p:{met:float(np.mean([v[p][agg][met] for v in ps])) for met in ['iou','dice']} for p in ps[0]};summary={met:stat([p[met] for p in patients.values()]) for met in ['iou','dice']};avgrows.append(dict(fit=fit,rule=rule,variant=variant,resolution=res,aggregation=agg,patients=patients,summary=summary))
     if rule=='N1':
      for met in ['iou','dice']:
       v=[runs[(fit,s)]['rules'][rule]['results'][variant][res]['summary'][agg][met]['mean'] for s in range(3)]
       seedvariation.append(dict(fit=fit,variant=variant,resolution=res,aggregation=agg,metric=met,seed_values_percent=[100*x for x in v],sd_seed_means_percent=100*float(np.std(v,ddof=1)),range_pp=100*(max(v)-min(v))))
# Saved baseline/head results; align patient aggregation, retaining different mapping/feature protocols explicitly.
base=ROOT/'audit/scratch/baseline';heads=json.loads((base/'heads.json').read_text());comparisons=[]
for key,x in [('Frozen direct K=8 224, seed'+str(s),json.loads((base/f'k8_c0_s{s}.json').read_text())) for s in range(3)]+[(k,v) for k,v in heads.items()]:
 v={met:stat([p['heldout'][met] for p in x['lopo'].values()]) for met in ['iou','dice']};comparisons.append(dict(method=key,iou=v['iou'],dice=v['dice'],global_majority=x['majority'],global_optimal=x['optimal'],protocol='train16 image fit; four-val-patient LOPO optimal mapping (other3 labels); patient-pooled macro224',checkpoint=x.get('checkpoint'),checkpoint_exists=Path(x['checkpoint']).is_file() if x.get('checkpoint') else None))
oldfinal=json.loads((ROOT/'audit/scratch/final_baseline/summary.json').read_text());b0=next(x for x in oldfinal['summary'] if x['k']==8 and x['mapping']=='majority')
def avg(fit,var,res='224',agg='pooled'):return next(x for x in avgrows if x['fit']==fit and x['rule']=='N1' and x['variant']==var and x['resolution']==res and x['aggregation']==agg)
best={fit:max(range(3),key=lambda s:runs[(fit,s)]['rules']['N1']['results']['F1']['224']['summary']['pooled']['iou']['mean']) for fit in ['inductive','transductive']}
write_json('results.json',dict(producer='publication/report_results.py',manifest_sha256=sha(OUT/'stageA_manifest.json'),all_configurations=allrows,seed_averaged_patient_results=avgrows,seed_metric_variation=seedvariation,failures=failures,comparisons=comparisons,b0_reference=b0,descriptive_best_F1_224_seed=best,best_status='Retrospective label-informed maximum only, not a chosen configuration or recommendation.'))
csvwrite('all_configurations.csv',allrows);csvwrite('per_patient.csv',perrows);csvwrite('failure_diagnostics.csv',failures)
# Publication figure: paired per-patient scores, averaged across all three seeds, never best-seed selection.
fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,fit in zip(axes,['inductive','transductive']):
 a,b=avg(fit,'hard')['patients'],avg(fit,'F1')['patients']
 for pat in a:ax.plot([0,1],[100*a[pat]['iou'],100*b[pat]['iou']],color='0.7',lw=.7,marker='o',ms=3)
 ax.plot([0,1],[100*avg(fit,v)['summary']['iou']['mean'] for v in ['hard','F1']],color='#174a65',marker='s',lw=2,label='Equal-patient mean')
 ax.set_xticks([0,1],['Hard K-means','F1']);ax.set_ylabel('Liver IoU (%)');ax.set_title(fit.title()+f' ({len(a)} patients)');ax.set_ylim(50,90);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=8)
fig.suptitle('Label-free named clusters: paired patient results at 224 pixels');fig.tight_layout();fig.savefig(P/'patient_results.png',dpi=220);fig.savefig(P/'patient_results.pdf');plt.close(fig)
# Eight previously fixed validation slices, seed0 display fixed in this script, not selected by score.
rr=json.loads((OUT/'image_records.json').read_text());fixed=json.loads((OLD/'figure_selection.json').read_text())['eight_indices'];truth=np.load(OLD/'scoring_masks_native.npy',mmap_mode='r');panels=np.load(P/'reproduction/inductive_s0_figure_masks.npz');fig,ax=plt.subplots(8,4,figsize=(12,23))
for row,i in enumerate(fixed):
 im=raw_image(rr[i]);ax[row,0].imshow(im,cmap='gray',vmin=0,vmax=1);ax[row,0].set_title(rr[i]['stem'],fontsize=9);ax[row,1].imshow(im,cmap='gray',vmin=0,vmax=1);ax[row,1].contour(truth[i],[.5],colors=['#d62728'],linewidths=.7);ax[row,1].set_title('Ground truth',fontsize=9)
 for col,var in [(2,'hard'),(3,'F1')]:
  pr=panels[f'N1_{var}_{i}'];v=scores(confusion(pr,truth[i],2),[1]);ax[row,col].imshow(im,cmap='gray',vmin=0,vmax=1)
  if pr.any():ax[row,col].contour(pr,[.5],colors=['#00bd79'],linewidths=.7)
  ax[row,col].set_title(f'{var}: native IoU {100*v["iou"]:.1f}%',fontsize=9)
 for a in ax[row]:a.axis('off')
fig.suptitle('Inductive seed0; N1 = N2 = N3; fixed historical slices',fontsize=13);fig.tight_layout(rect=(0,0,1,.982));fig.savefig(P/'fixed_slices.png',dpi=180);fig.savefig(P/'fixed_slices.pdf');plt.close(fig)
# Main report.
lines=['# Frozen BiomedCLIP CT liver segmentation: final label-free naming results','',
'## Result and scope','',
'All six sealed fits were re-evaluated on CPU with the existing scoring/prediction code. Every numerical result, per-slice score, diagnostic ranking and reference agreement reproduced exactly (elapsed time excluded). No fitting, training, parameter change, source edit or replacement of a sealed artifact was performed. N1, N2 and N3 select the same cluster within every fit, and all selections agree with both saved label-assisted reference mappings. Naming removes the need for patient annotations at the current naming/inference stage; it does not change the masks relative to those references.',
'',f"Across seeds, the four-patient inductive F1 estimate is **IoU {fmt(avg('inductive','F1')['summary']['iou'])}% and Dice {fmt(avg('inductive','F1')['summary']['dice'])}%**. The twenty-patient transductive estimate is **IoU {fmt(avg('transductive','F1')['summary']['iou'])}% and Dice {fmt(avg('transductive','F1')['summary']['dice'])}%**. These are primary224, patient-pooled metrics; SD is across patients after averaging each patient's seed scores, not a confidence interval or three independent cohorts.",
'', '## Fixed method and statistical definitions','',
'Frozen BiomedCLIP ViT-B/16, final LayerNorm patch tokens then L2 normalization, no centering, body-cropped448 input and 28×28 patch grid; K=8 saved centres for seeds 0/1/2. CT[-150,250] HU window, grayscale replicated to3 channels, BiomedCLIP normalization once. Inductive centres use 16 training patients/1869 slices; only the four validation patients/472 slices are scored. Transductive centres use all 20 patients/2341 slices and those same 20 are scored. No refit was required.',
'', 'N1 uses 200 nearest eligible body patches per centre without requiring prior hard-cluster membership, 112-pixel native crops resized to 224, joint-space BiomedCLIP image/text embeddings, fixed 15 prompts, and maximum liver z-score. N2 uses rawHU>-500, disk 5 opening, largest component and hole filling; x<.5,y<.6,mean HU 40..180 candidates ranked by body-patch area. N3 accepts exact N1/N2 agreement only. Full operational parameters are in the sealed ../PARAMETERS.json and ../stageA_manifest.json.',
'', 'Hard predictions use the inherited nearest-grid inverse crop. F1 uses existing per-fit softmax distance temperature, bilinear inverse crop, guided filter radius 8/epsilon .01, threshold .5, largest 2D component, hole filling and disk 3 closing. F1 is not merely soft assignment: these post-processing steps are part of its fixed definition. No F2 median filter is used. Stored per-fit temperatures are listed in audit_verification.json.',
'', 'Per-patient pooled IoU = sum(TP)/(sum(TP)+sum(FP)+sum(FN)); Dice =2sum(TP)/(2sum(TP)+sum(FP)+sum(FN)). Cohort summaries weight patients equally. Mean-slice aggregation averages slice scores within each patient, then summarizes patients equally. The inherited scorer assigns 0 to empty-prediction/empty-truth slices; both aggregations preserve that convention. SD uses ddof=1. No pixel/slice count is treated as the number of independent patients. No significance test or confidence interval is claimed.',
'', '## Main results: primary224, patient-pooled percentages','',
'Agreement is exact equality of the named singleton with the majority / optimal reference subset for each scored patient. All three rules are shown; none was selected by score. Inductive reference mappings use train16 labels; transductive reference mappings use other19 labels in LOPO. These references never enter StageA.']
main=[]
for (fit,seed),run in runs.items():
 for rule in ['N1','N2','N3']:
  r=run['rules'][rule];a=r['agreement']['224'];agreement=' / '.join(str(sum(v[q] for v in a.values()))+'/'+str(len(a)) for q in ['majority','optimal']);h=r['results']['hard']['224']['summary']['pooled'];f=r['results']['F1']['224']['summary']['pooled']
  main.append([fit,seed,rule,r['selected_cluster'],agreement,fmt(h['iou']),fmt(h['dice']),fmt(f['iou']),fmt(f['dice'])])
lines+=['',table(['Protocol','Seed','Rule','Cluster','Agreement M/O','Hard IoU','Hard Dice','F1 IoU','F1 Dice'],main),
'', 'All three rules abstain in 0/3 inductive fits and0/3 transductive fits; N3 has 0/6 abstentions overall. No abstaining case occurred, so performance of the abstention mechanism on failure cases is NOT VERIFIED.',
'', '## Mean across seeds, both scales and both aggregations','',
'Average each patient\'s three seed scores first, then calculate mean/sampleSD/range across patients. This is not an ensemble mask. Identical N1/N2/N3 and reference-majority/reference-optimal results are represented once here; every row is retained in all_configurations.csv and the supplementary tables.']
rows=[]
for fit in ['inductive','transductive']:
 for res in ['224','512']:
  for agg in ['pooled','mean_slice']:
   for var in ['hard','F1']:
    r=avg(fit,var,res,agg);rows.append([fit,res,agg,var,fmt(r['summary']['iou']),fmt(r['summary']['dice'])])
lines+=['',table(['Protocol','Scale','Aggregation','Prediction','IoU mean ± SD [min–max]','Dice mean ± SD [min–max]'],rows),
'', '## Descriptive maxima: explicitly label-informed, not selected configurations','']
for fit in ['inductive','transductive']:
 s=best[fit];v=runs[(fit,s)]['rules']['N1']['results']['F1']['224']['summary']['pooled'];lines.append(f"- {fit.title()}: seed{s}, F1 IoU {fmt(v['iou'])}%, Dice {fmt(v['dice'])}%. This retrospective maximum uses evaluation labels. It is not a recommended seed, a confirmatory estimate, or permission to tune. N1/N2/N3 tie exactly.")
lines+=['', '## Seed stability: IDs versus actual masks','',
'Cluster IDs have no shared meaning across fits. We align complete 8-cluster partitions with label-free Hungarian matching of patch co-occurrence, and separately compare the actual named binary hard/F1 masks on all evaluated slices. Thus permutation equivalence is tested rather than inferred from similar IoU. Inter-seed Dice below compares predictions with predictions and uses no ground truth.',
'']
srows=[]
for fit,z in stability['results'].items():
 for pair in z['pairs']:
  h=pair['predictions']['hard']['512']['pooled'];f=pair['predictions']['F1']['512']['pooled'];pa=pair['partition']['evaluated_images'];srows.append([fit,str(pair['seeds']),f"{100*pa['aligned_agreement']:.6f}",pa['selected_ids_correspond'],f"{100*h['binary_dice']:.6f}",f"{100*f['binary_dice']:.6f}",f"{h['identical_slices']}/{h['slices']}",f"{f['identical_slices']}/{f['slices']}",f"{100*h['disagreement_fraction']:.6f}",f"{100*f['disagreement_fraction']:.6f}"])
lines+=[table(['Protocol','Seeds','Aligned partition agreement %','Named IDs correspond','Hard inter-seed Dice %','F1 inter-seed Dice %','Identical hard slices','Identical F1 slices','Hard pixel disagreement %','F1 pixel disagreement %'],srows),
'', 'Inductive seeds produce exactly identical hard liver masks on all 472 validation slices; native F1 differs by only 2–20 pixels per seed pair across the entire evaluation set. Their selected IDs (2,3,4) denote the same liver region, although the complete 8-cluster partition is not bit-identical after permutation. Transductive seed1 is not merely an ID permutation of seeds 0/2: aligned partition agreement is about 87.44%, hard liver-mask inter-seed Dice about 98.02%, and F1 inter-seed Dice about 98.38%. Seeds0/2 are much closer. Thus semantic naming is stable, but all transductive masks are not equivalent. A partition agreement below 100% rules out an exact global relabeling. Different selected IDs can still correspond to the same liver-like region, but mask equality is asserted only when differing-pixel counts are zero. F1 may amplify small centre/temperature differences through connected-component decisions. No seed was chosen to improve these comparisons.',
'',table(['Protocol','Prediction','Metric','Seed0 /1 /2 means %','SD of seed means','Range pp'],[[r['fit'],r['variant'],r['metric'],' / '.join(f'{v:.6f}' for v in r['seed_values_percent']),f"{r['sd_seed_means_percent']:.6f}",f"{r['range_pp']:.6f}"] for r in seedvariation if r['resolution']=='224' and r['aggregation']=='pooled']),
'', '## Baseline comparison: separate matched and historical comparisons','',
'**Exactly matched448 baseline:** reference-majority and reference-optimal yield exactly the same masks/scores as label-free naming on these fits. The naming gain in IoU is 0 percentage points; the gain is avoiding annotation-based semantic assignment. F1 versus hard K-means changes post-processing, not naming.',
'',table(['Protocol','Direct448 hard IoU','F1 IoU','F1 minus hard IoU pp','Direct448 hard Dice','F1 Dice'],[[fit,f"{100*avg(fit,'hard')['summary']['iou']['mean']:.3f}",f"{100*avg(fit,'F1')['summary']['iou']['mean']:.3f}",f"{100*(avg(fit,'F1')['summary']['iou']['mean']-avg(fit,'hard')['summary']['iou']['mean']):.3f}",f"{100*avg(fit,'hard')['summary']['dice']['mean']:.3f}",f"{100*avg(fit,'F1')['summary']['dice']['mean']:.3f}"] for fit in ['inductive','transductive']]),
'', '**Historical224-input baseline and STEGO heads:** the table below uses the same four validation patients,224 scoring masks and equal-patient pooled-pixel definitions. Historical mapping is optimal LOPO over the other three validation patients. Current names are label-free from train images; grid/crop/upsampling/post-processing differ. This is a descriptive pipeline comparison, not a controlled attribution of gains to naming or to one component. The full batch600 checkpoint was selected using labels; that selection remains optimistic.',
'',table(['Method','Patient-mean IoU ± SD [range]','Patient-mean Dice ± SD [range]','Global majority IoU/Dice','Global optimal IoU/Dice'],[[x['method'],fmt(x['iou']),fmt(x['dice']),pd(x['global_majority']),pd(x['global_optimal'])] for x in comparisons]),
'',f"Historical all 20 transductive direct224-input K=8 baseline: IoU {fmt(b0['iou'])}%, Dice {fmt(b0['dice'])}%, other19 LOPO mapping. Current448 hard improves the reported patient mean by {100*(avg('transductive','hard')['summary']['iou']['mean']-b0['iou']['mean']):.3f} IoU points; F1 by {100*(avg('transductive','F1')['summary']['iou']['mean']-b0['iou']['mean']):.3f} points. These are combined historical pipeline changes on a reused cohort, not an untouched-test effect or an isolated naming improvement.",
'', 'Saved STEGO scores are reused, not retrained or newly inferred. Checkpoint existence and source paths are in results.json; a missing file would not erase the saved result but would prevent independent checkpoint rerunning. New224/native mean-slice results for historical STEGO are NOT VERIFIED here; no false equivalence with current F1 is implied.',
'', '## Label-use and leakage audit','',
table(['Stage','Labels accessed?','Purpose / evidence'],[['Original StageA','NO, in the recorded guarded process','Images, raw DICOM HU, frozen features/centres, crop/image/text embeddings and body statistics; no masks or saved semantic mappings. Metadata contained unused mask filenames/hashes.'],['Current stability/naming recheck','NO','Saved embeddings/statistics rederive names; actual prediction comparisons and partition alignment; no scoring labels.'],['Original and repeated StageB','YES','IoU/Dice, cluster ranking, overlays, reference agreement; inductive stored train-label references and transductive other19 LOPO references are comparisons only.'],['Historical research/development','YES','K=8,448 resolution, CT window and F1 parent selection were previously informed by these patients\' labels.'],['Descriptive best-seed summary','YES','Retrospective reporting only; no seed/configuration chosen for deployment or new evaluation.']]),
'',f"Manifest SHA256 `{audit['manifest_sha256']}`; all{audit['sealed_file_count']} sealed files match. Initial seal {audit['initial_seal_utc']}; final seal {audit['final_seal_utc']}; StageB started {audit['stageB_start_utc']} — {audit['seconds_between_final_seal_and_B']:.6f} seconds after the final seal. Timestamps are quoted in UTC as stored.",
'', 'The failed first attempt stopped because an extra hard-assignment restriction left only 13 eligible patches for one cluster. The correction removed that unintended restriction to implement the declared all-body-patches rule. The successful log restarts GEOMETRY at 0 and traverses the dataset, then repeats nearest-patch selection, embeddings and naming before sealing. The frozen code recomputes those body statistics directly from raw DICOM. No corrected parameters were chosen by a score sweep.',
'', '**Historical evidentiary limit:** logs, frozen code, the read guard, the zero-label-read ledger and execution notes support pre-scoring correction and label-free StageA. They are local self-generated records, not an independent OS-level audit. A complete access ledger for the failed first attempt and proof of absence of any external/cognitive score influence are **NOT VERIFIED**. No claim is made that historical labels were never seen. After sealing, all 42 artifact hashes still match and rederived selections equal the sealed ones; no subsequent change to those naming outputs was found.',
'', 'Label-free inference means new image predictions need no masks. Label-free naming means these cluster IDs were assigned using text/anatomy without masks. Neither phrase means the pretrained vision-language model had no image-text supervision, nor that the historical research process was fully unsupervised.',
'', '## Failure modes and limits','',
'Native-resolution diagnostics below use labels only to describe errors. Tiny nonempty slices have fewer than 2000 liver pixels; empty slices are reported separately. The threshold is a reporting stratum from the prior protocol, not a pipeline parameter.',
'',table(['Protocol','Seed','Prediction','Empty slices','Empty with any FP','Mean FP pixels/empty slice','Tiny slices','Tiny mean-slice IoU %','Other nonempty slices','Other mean-slice IoU %'],[[x['fit'],x['seed'],x['variant'],x['empty_slices'],x['empty_any_fp'],(f"{x['empty_mean_fp']:.2f}" if x['empty_mean_fp'] is not None else 'N/A'),x['tiny_slices'],f"{x['tiny_mean_slice_iou']:.3f}",x['large_slices'],f"{x['large_mean_slice_iou']:.3f}"] for x in failures]),
'', 'There are no empty-ground-truth slices in either evaluated set; the empty-slice false-positive statistic is N/A, not zero false positives. Specificity on liver-absent slices is NOT VERIFIED. Tiny-liver failures are severe: for inductive seed0, native mean-slice IoU falls from 2.731% hard to 0.093% F1 on 30 slices with 1–1999 liver pixels. The pooled-pixel improvement is therefore not a uniform improvement on all anatomy sizes. The hard liver-like cluster still spills into other organs and has28×28-grid boundary artifacts. F1 often removes disconnected spill and fills holes, but largest-component filtering can discard genuine disconnected liver and morphological closing cannot restore missing information. N2 assumes the documented native axial orientation. N1 margins are descriptive scores, not calibrated uncertainty. N3 was never tested on a disagreement case in these six fits. Twenty CT patients from one acquisition cohort, only four in the inductive evaluation, and repeated label-informed development limit generalization. Independent-cohort performance, clinical suitability, and significance of small seed differences are NOT VERIFIED.',
'', '## Figures and reproducibility','',
'![Paired patient results](patient_results.png)','',
'Figure1. Primary224 pooled-per-patient IoU, each patient averaged across seeds. Grey lines are patients; blue is the equal-patient mean. No confidence interval implied. [Vector PDF](patient_results.pdf).','',
'![Fixed slices](fixed_slices.png)','',
'Figure2. The eight previously fixed validation slices; inductive seed0 is displayed by a fixed illustrative convention, not because it scored highest. Ground truth red; named hard/F1 contours green. All three rules are identical here. Slice selection was historically label-informed. [Vector PDF](fixed_slices.pdf).','',
'Producers: reproduce.py re-executes the sealed scorer with only output destinations redirected; audit_record.py verifies seals and rederives names without labels; stability.py compares actual predictions without labels; report_results.py computes all aggregates, tables and figures. Outputs: reproduction_verification.json, audit_verification.json, stability.json, results.json, all_configurations.csv, per_patient.csv, failure_diagnostics.csv and SUPPLEMENTARY_TABLES.md. Original evidence: ../NAMING.md, ../EXECUTION_NOTES.md, ../stageA_manifest.json, ../stageA_ledger.json and the sealed outputs. All source/data/cache/checkpoint files are read-only in this workflow.',
'', '## Publication-ready interpretation','',
f"On this repeatedly studied CHAOS CT cohort, frozen BiomedCLIP features with K=8 clustering support annotation-free semantic cluster naming using either fixed text prompts or a fixed anatomical/HU prior. Both rules and their agreement rule identify the same liver-like cluster in all six saved fits and reproduce the label-assisted reference predictions. With the fixed F1 post-processing, the seed-averaged inductive result is {100*avg('inductive','F1')['summary']['iou']['mean']:.2f}% liver IoU and {100*avg('inductive','F1')['summary']['dice']['mean']:.2f}% Dice across four patients; the transductive result is {100*avg('transductive','F1')['summary']['iou']['mean']:.2f}% IoU and {100*avg('transductive','F1')['summary']['dice']['mean']:.2f}% Dice across 20 patients. This pooled-pixel estimate coexists with severe failure on tiny-liver slices and untested liver-absent-slice specificity. The scientific advance over the saved label-assisted448 pipeline is removal of label-based cluster naming, with no loss of its recorded segmentation scores—not an additional accuracy gain from naming. F1 improves the direct hard-clustering scores on this cohort. These results do not establish fully unsupervised historical development, independence from pretrained text supervision, performance on an untouched test cohort, or clinical robustness. Historical K/resolution/window/post-processing choices and retrospective best-seed reporting remain label-informed; no further tuning or head training was performed."]
(P/'FINAL_RESULTS.md').write_text('\n'.join(lines)+'\n')
sup=['# Supplementary tables','', 'All values are percentages. Source: exact re-evaluation via reproduce.py; aggregation by report_results.py. SD across patients. Each fit/rule/seed/prediction/scale/aggregation is retained, including reference mappings. CSVs contain full precision.','',table(['Fit','Seed','Rule','Prediction','Scale','Aggregation','IoU mean ± SD [min–max]','Dice mean ± SD [min–max]'],[[r['fit'],r['seed'],r['rule'],r['variant'],r['resolution'],r['aggregation'],f"{r['iou_mean']:.6f} ± {r['iou_std']:.6f} [{r['iou_min']:.6f}–{r['iou_max']:.6f}]",f"{r['dice_mean']:.6f} ± {r['dice_std']:.6f} [{r['dice_min']:.6f}–{r['dice_max']:.6f}]"] for r in allrows]),'', '## Per-patient results','',table(['Fit','Seed','Rule','Prediction','Scale','Aggregation','Patient','Slices','IoU','Dice'],[[r[k] if k not in ['iou','dice'] else f'{r[k]:.6f}' for k in ['fit','seed','rule','variant','resolution','aggregation','patient','slices','iou','dice']] for r in perrows])]
(P/'SUPPLEMENTARY_TABLES.md').write_text('\n'.join(sup)+'\n')
print('REPORT COMPLETE',elapsed(),len(allrows),len(perrows),flush=True)
