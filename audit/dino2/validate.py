from common import *
import subprocess
v=json.loads((OUT/'weight_verification.json').read_text());assert v['passed']
for arm,r in v['arms'].items():
 assert sha(r['file'])==r['expected_sha256']
 assert sha(Path(r['file']).parent/'config.json')==r['config_sha256']
 assert sha(Path(r['file']).parent/'preprocessor_config.json')==r['preprocessor_sha256']
 if arm=='D2':assert sha(r['source_bin']['file'])==r['source_bin']['expected_sha256']
sel=json.loads((OUT/'pair_selectivity.json').read_text());previous=json.loads((ROOT/'audit/dino/pair_selectivity.json').read_text())['results']['BiomedCLIP_B2_448']
assert sel['results']['BiomedCLIP_B2_448']==previous,'BiomedCLIP matched reference drift'
assert np.array_equal(np.load(OUT/'pair_a.npy'),np.load(ROOT/'audit/dino/pair_a.npy'))
assert np.array_equal(np.load(OUT/'pair_ta.npy'),np.load(ROOT/'audit/dino/pair_ta.npy')) and np.array_equal(np.load(OUT/'pair_tb.npy'),np.load(ROOT/'audit/dino/pair_tb.npy'))
norm_errors={}
for arm in ['D2','D3']:
 if not (OUT/f'{arm}_extract.json').exists():continue
 c=json.loads((OUT/f'{arm}_checks.json').read_text());assert c['model_eval'] and c['all_parameters_frozen'] and c['deterministic_exact'] and c['final_LayerNorm_hook_exact']
 assert all(not c['strict_loading_info'].get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
 f=np.load(OUT/f'{arm}_features.npy',mmap_mode='r');assert f.shape==(2341,784,768);error=0.
 for start in range(0,2341,32):
  x=np.asarray(f[start:start+32]);assert np.isfinite(x).all();error=max(error,float(np.abs(np.linalg.norm(x,axis=-1)-1).max()))
 assert error<1e-5;norm_errors[arm]=error
 figure_time=(OUT/f'{arm}_figures.json').stat().st_mtime
 for k in [8,2]:
  for seed in [0,1,2]:
   path=OUT/f'{arm}_k{k}_s{seed}_scores.json';assert path.stat().st_mtime>figure_time
   r=json.loads(path.read_text());assert r['arm']==arm
   for res in ['224','512']:
    folds=r['results'][res]['folds'];assert len(folds)==20
    for pat,fold in folds.items():assert pat not in fold['mapping_patients'] and len(fold['mapping_patients'])==19
state=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),diff=subprocess.check_output(['git','diff'],cwd=ROOT,text=True),status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True));assert state['diff']==''
dump('validation.json',dict(producer='validate.py',backbone_files_unchanged=True,exact_BiomedCLIP_selectivity_reference=True,exact_D1_sampling_and_label_indices=True,L2_errors=norm_errors,all_L0PO_mappings_exclude_heldout=True,figures_before_scoring=True,source_state=state))
print('VALIDATED',norm_errors)
