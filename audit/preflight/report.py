"""Render the preflight evidence without accessing annotations or experiment outputs for scoring."""
from diagnose import *
import inspect
from collections import Counter
o=read(HERE/'objective.json'); init=read(HERE/'initialization.json'); trajectories=read(HERE/'trajectories.json')
sample=read(HERE/'sample.json'); checks=read(HERE/'extract_checks.json')
div={name:read(HERE/file) for name,file in [('Final (12)','diversity_final.json'),('Block 9','diversity_block9.json'),('Block 11','diversity_block11.json')]}
sealed=read(EXP/'logs/preflight.json')
hash_checks={name:sha(EXP/name)==sealed['source_sha256'][name] for name in ['model.py','reference_loss.py','train.py']}
assert all(hash_checks.values())
fit=read(ROOT/'audit/final_pipeline/inductive_seed0_fit.json')
assert fit['fit_labels_used'] is False and fit['centers_sha256']==sha(ROOT/'audit/final_pipeline/inductive_seed0_centers.npy')
assert checks['backbone_sha256']==sealed['backbone_sha256']
dump('verification.json',dict(source_matches_sealed=hash_checks,backbone_matches_sealed=True,center_matches_fit=True,
 config_sha256=sha(EXP/'configs/locked.json'),threads=torch.get_num_threads(),labels_opened=False))

lines=[]
def put(s=''):
    lines.append(s)
    if s and not s.startswith('|'): lines.append('')
def table(headers,rows):
    put('| '+' | '.join(headers)+' |');put('| '+' | '.join(['---']*len(headers))+' |')
    for r in rows:put('| '+' | '.join(map(str,r))+' |')
    put()
put('# Label-free preflight: sealed 448 / 28×28 STEGO head')
put('All numerical diagnostics are produced by `audit/preflight/diagnose.py` (`core` and `extract`); this report and the derived tables/figures are produced by `audit/preflight/report.py`. No gradients, optimizer steps, label/mask files, or validation images were used. CPU only, `torch.set_num_threads(2)`. Existing experiment sources, caches and checkpoints were read-only. New intermediate arrays and reports are confined to `audit/preflight/`.')
put()
put('## 1. Pinned configuration and implementation')
table(['Parameter','Exact value','Evidence (repository-relative file:line)'],[
 ['Shifts: intra / kNN / random cross-image','0.18 / 0.12 / 0.46','experiments/stego_biomedclip/configs/locked.json:24'],
 ['Loss weights: intra / kNN / random','0.67 / 0.25 / 0.63; overall 1.0','experiments/stego_biomedclip/configs/locked.json:27'],
 ['LR / optimizer / schedule','0.0005 / Adam / none','experiments/stego_biomedclip/configs/locked.json:12; experiments/stego_biomedclip/train.py:33'],
 ['Batch / budget','16 / 5000 batches','experiments/stego_biomedclip/configs/locked.json:10'],
 ['Dropout','Dropout2d(p=0.1), independent calls for both branches and feature signal','experiments/stego_biomedclip/model.py:11; :18; :21'],
 ['Head','Conv1×1(768→70) + [Conv1×1(768→768), ReLU, Conv1×1(768→70)]; biases enabled','experiments/stego_biomedclip/model.py:12'],
 ['Initialization','PyTorch Conv2d defaults: kaiming_uniform_(a=sqrt(5)); bias uniform ±1/sqrt(fan_in). All three convolutions have fan_in=768. No manual head initialization.','experiments/stego_biomedclip/model.py:9; '+init['reset_parameters_source']+':182'],
 ['Centering','Pointwise target mean removed, global mean restored; normalized feature correlation','experiments/stego_biomedclip/reference_loss.py:54'],
 ['Clamping','Code cosine clamped below at 0; stabalize=False (no 0.8 cap)','experiments/stego_biomedclip/reference_loss.py:61; configs/locked.json:20'],
 ['Spatial sampling / negatives','11×11 uniform grid-sampled coordinates; 5 non-self batch permutations','experiments/stego_biomedclip/configs/locked.json:17; reference_loss.py:91; :108'],
 ['kNN selection','Uniform rank among first 7 of stored 30, train-only','experiments/stego_biomedclip/train.py:26; :57'],
 ['Other losses','Geometric augmentation alignment 0; actual objective sums only the three correspondence terms','experiments/stego_biomedclip/configs/locked.json:31; model.py:28'],
 ['Monitor rule','Every 50 batches; cosine >0.99 OR std <0.01 for 3 consecutive checks. Reports collapse; does not early-stop.','experiments/stego_biomedclip/train.py:66; :77; configs/locked.json:44']])
put('Pinned configuration SHA256: `'+sha(EXP/'configs/locked.json')+'`. Imported head/loss/training source hashes match the sealed preflight hashes (`verification.json`); backbone weight SHA256 is `'+checks['backbone_sha256']+'`. The sealed training used 4 CPU threads; these diagnostics use the requested 2. No configuration was changed.')
put()
put('## 2. Sampling and loss comparison')
put(f"Fixed RNG seed {sample['seed']}; {sample['sample_count']} uniformly sampled slices without replacement from {sample['train_population']} train slices, covering all {len(sample['patient_counts'])} train patients. Sample IDs and patient counts: `sample.json`; neighbor IDs: `pair_plan.json`. Frozen final-block 448 tokens are read only at training indices from `audit/improve/B2_448_features.npy`. A fresh replay of all 200 training images reproduces the cached final tokens with maximum absolute error {checks['final_cached_max_error']}; body crop boxes agree: {checks['body_boxes_agree']}.")
put('The exact imported `ContrastiveCorrelationLoss.forward/helper` samples and centers the affinities. For constant code and shift zero, its negative loss recovers the affinity without duplicating the centering formula. Same-image includes the helper’s diagonal/self pairs. kNN neighbors come from the full train-only cache, selected uniformly among its first 7 ranks; random negatives use the actual 5 non-self batch permutations. Batch size is 16, with one final batch of 8; centering is recomputed in each helper call exactly as in the loss. Counts are pair draws, not independent patients: '+str(o['affinity_count'])+' for intra/kNN/random, respectively.')
put('Feature dropout is disabled for these normalized-token comparisons, matching the fixed train monitor’s evaluation mode. Constant codes are the same nonzero 70-D vector everywhere; random codes are independent normalized Gaussian 70-D fields; cluster codes are one-hot K=8 assignments padded to 70-D from the saved train-only seed-0 448 fit (`audit/final_pipeline/inductive_seed0_centers.npy`, digest in `pair_plan.json`). No fit or label mapping is performed. The loss bilinearly samples all codes, including the one-hot grids, before its code normalization. Identical sampling/permutations are replayed for every code type and shift group; the loss weights remain pinned for all comparisons.')
put()
table(['Shift group','Intra / kNN / random shifts','Attract intra %','Attract kNN %','Attract random %','Constant total','Random total','K8-code total'],[
 [name,' / '.join(f'{s:.8f}' for s in g['shifts']),*[f'{a*100:.3f}' for a in g['attraction']],*[f"{g['losses'][kind]['total']:.6f}" for kind in ['constant','random','cluster']]] for name,g in o['groups'].items()])
table(['Shift group','Code','Intra loss (unweighted)','kNN loss (unweighted)','Random loss (unweighted)','Weighted total'],[
 [name,kind,*[f'{v[k]:.8f}' for k in ['pos_intra','pos_inter','neg_inter','total']]] for name,g in o['groups'].items() for kind,v in g['losses'].items()])
put('Lower is better. At the pinned shifts, constant codes beat K8 codes by '+f"{o['groups']['Pinned']['losses']['cluster']['total']-o['groups']['Pinned']['losses']['constant']['total']:.8f}"+' loss units. This is a direct comparison of candidate representations, not a proof that constant codes are the unique/global optimum. C, D and E remove this constant-over-K8 advantage on the same sample; E nevertheless gives K8 codes a positive objective. These measurements cannot establish semantic quality or guarantee stable optimization.')
put('Historical shift sources: `audit/CROSSPATIENT.md:73` (C), `audit/FIXES4.md:20` (D/E). Their earlier attraction targets came from another resolution/sample; they are not assumed to transfer. Here the random-attraction rates for D/E are about 12.4%/6.9%, not the historical 10%/5%.')
put()
put('Label-free quantiles on this exact sample (diagnostic thresholds only; not applied):')
table(['Desired attraction','Intra shift','kNN shift','Random shift'],[[f'{(1-float(q))*100:g}%',*[f'{x:.8f}' for x in o['quantiles'][q]]] for q in ['0.75','0.8','0.85','0.9','0.95']])
put('These quantiles are heuristics, not paper-prescribed settings. The measured C/D/E comparisons support reduced random attraction and selective positives as candidates for removing constant-code preference; no universal attraction fraction prevents collapse.')
put()
put('## 3. Feature diversity')
put('All sources use the same 200 train slices and all 784 patch vectors per image. Blocks are 1-indexed (9, 11, final=12), each passed through the model’s shared final LayerNorm, with CLS excluded, then patch L2. The project extractor is imported; its existing `get_intermediate_layers` call is intercepted only to request indices [8,10,11] and returns the final tensor back to the unchanged extractor (`src/modules.py:210`). Preprocessing functions are AST-imported unchanged from `audit/final_pipeline/segment_ct.py:23`, :33 and :41 to avoid that file’s global side effects. No mask is used for the body crop.')
put('“Patch-pair cosine” is the exact mean across every pair of patches in different images: mean dot product of unnormalized image means of unit patch vectors. “Normalized-GAP cosine” L2-normalizes those image means first and measures image-level similarity; these are different quantities. Channel std uses all 156,800 tokens with sample variance. PCA uses the full centered 768×768 covariance (for statistical analysis only); effective rank = exp(entropy of all normalized nonnegative eigenvalues).')
table(['Source','Patch-pair cosine','Normalized-GAP cosine','Channel std mean [min,max]','PC1 %','PC1–10 %','PC1–50 %','Effective rank'],[
 [name,f"{v['patch_pair_cosine']:.6f}",f"{v['normalized_gap_cosine']:.6f}",f"{v['channel_std_mean']:.6f} [{v['channel_std_min']:.6f}, {v['channel_std_max']:.6f}]",f"{v['pc1']*100:.2f}",f"{v['pc10']*100:.2f}",f"{v['pc50']*100:.2f}",f"{v['effective_rank']:.3f}"] for name,v in div.items()])
put('Every per-channel std and all 768 explained-variance ratios are in `diversity_final.json`, `diversity_block9.json`, and `diversity_block11.json`; spectrum figure: `pca_spectrum.png`. Full extraction checks: `extract_checks.json`; identical-call max error '+str(checks['deterministic_max_error'])+', trainable backbone parameters '+str(checks['trainable_parameters'])+', extraction seconds '+f"{checks['seconds']:.3f}"+'. The final-token diversity is nonzero; near-unit head cosine is not already present in the patch-pair backbone representation.')
put()
put('## 4. Initialization and observed trajectories')
put('Initialization uses the sealed monitor’s exact 16 fixed train images (one per patient), head construction after torch.manual_seed(seed), eval mode, no gradients. Code std and cross-image cosine match `experiments/stego_biomedclip/train.py:72–76`: L2-normalize 70-D codes; average per-image/per-channel spatial std; average dot products of image-mean unit patch vectors across different images. No learned state is loaded. Monitor IDs are in `initialization.json`.')
table(['Fresh head seed','Code std','Cross-image patch-pair cosine'],[[r['seed'],f"{r['code_std']:.8f}",f"{r['cross_image_cosine']:.8f}"] for r in init['rows']])
put('Existing logs only, with every monitor record preserved below. `trajectories.json` records the source SHA256, byte offsets and record ordinals; no final checkpoint was re-run. Seed 1 contains '+str(trajectories['1']['records'])+' JSON records and '+str(len(trajectories['1']['monitor_rows']))+' monitor records, including duplicate batches and interleaved resumed streams. The resume record explicitly says optimizer and RNG state were not restored (`experiments/stego_biomedclip/logs/seed1_resume.json:5`). Therefore a unique uninterrupted seed-1 trajectory after the first resume is UNVERIFIED; the source log alone cannot attribute duplicate rows to a unique process/checkpoint. Both final batch-5000 observations are retained. This does not affect the seed-1 collapse already observed before the first resume.')
for seed,x in trajectories.items():
    put('### Seed '+seed+' — every recorded monitor')
    table(['Log record','Batch','Code std','Cross-image cosine','Bad checks','Collapse flag'],[
      [r['record_ordinal'],r['batch'],f"{r['code_std']:.9f}",f"{r['cross_image_cosine']:.9f}",r['consecutive_bad_checks'],r['collapse_detected']] for r in x['monitor_rows']])
put('Seed 0 and seed 1 both first meet the bad-check condition at batch 50 and first set the three-consecutive-check collapse flag at batch 150 (`trajectories.json`, original log rows 50/100/150). Their initialized heads were spatially variable before optimization; the logged training rapidly removed this variation. `trajectories.png` plots all recorded checks; seed-1 duplicates are deliberately retained as separate points.')
put()
put('## 5. Evidence, limits and reproduction')
put('Commands (working directory: repository root):\n```sh\npython3 -B audit/preflight/diagnose.py core\npython3 -B audit/preflight/diagnose.py extract\npython3 -B audit/preflight/report.py\n```')
put('Access ledgers: `core_access.json`, `extract_access.json`, `report_access.json`. The runtime rejects label/mask paths, validation-image paths, network access, and writes outside this output directory. The shared cached feature file contains train and validation images; only explicit train indices were materialized. The image-only metadata includes all sample descriptors; no annotation or validation image was opened. No training, experiment edits, cache regeneration, or installs occurred.')
put('UNVERIFIED: the full stochastic training objective with feature dropout enabled was not estimated here; comparisons use normalized frozen tokens/dropout-off as requested and as used in the fixed monitor. Initialization diagnostics do not isolate the contribution of bias versus each branch. No labels were opened, so anatomical selectivity or segmentation performance of any layer/shift is unverified. No stable alternative training or guarantee follows from these loss comparisons. A unique seed-1 continuation after resume requires process-specific logs/checkpoint lineage and restored optimizer/RNG evidence; current combined logs are insufficient. Confidence intervals over patient resampling were not computed; pair draws are highly dependent.')
put('Prediction: the pinned loss strongly favors constant codes over the tested clustered codes and predicts collapse, consistent with the already-recorded seed-0/1 trajectories. C/D/E-like reductions in attraction remove that measured preference here, but no attraction percentage alone guarantees non-collapse or liver segmentation.')
(HERE/'PREFLIGHT.md').write_text('\n'.join(lines)+'\n')

import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for name,v in div.items():
    p=np.array(v['explained_variance_ratio'])
    axes[0].semilogy(np.arange(1,769),p,label=name)
    axes[1].plot(np.arange(1,769),p.cumsum(),label=name)
axes[0].set(xlabel='Principal component',ylabel='Explained variance fraction',title='Full PCA spectrum — 200 train slices')
axes[1].set(xlabel='Number of components',ylabel='Cumulative explained variance',title='Cumulative PCA variance')
for ax in axes:ax.legend();ax.grid(alpha=.2)
fig.savefig(HERE/'pca_spectrum.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for seed,x in trajectories.items():
    r=x['monitor_rows'];steps=[v['batch'] for v in r]
    axes[0].scatter(steps,[v['code_std'] for v in r],s=8,label='Seed '+seed)
    axes[1].scatter(steps,[1-v['cross_image_cosine'] for v in r],s=8,label='Seed '+seed)
axes[0].set(yscale='log',xlabel='Logged batch',ylabel='Normalized code spatial std',title='Recorded code diversity')
axes[1].set(yscale='symlog',xlabel='Logged batch',ylabel='1 − cross-image cosine',title='Recorded distance from constant codes')
for ax in axes:ax.legend();ax.grid(alpha=.2)
fig.savefig(HERE/'trajectories.png',dpi=180);plt.close(fig)
dump('report_access.json',dict(threads=torch.get_num_threads(),labels_allowed=False,opened_read_paths=sorted(READS),script_sha256=sha(__file__)))
print('Wrote',HERE/'PREFLIGHT.md')
