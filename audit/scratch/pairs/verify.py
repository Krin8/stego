from common import *
import subprocess
setup=json.loads((OUT/'setup.json').read_text());p=json.loads((OUT/'purity.json').read_text())
for key,row in p['summary'].items():
 for kind in ['random_unfiltered','attracted']:
  r=row[kind];assert sum(r['counts_BB_BL_LB_LL'])==r['pairs']
  if r['pairs']:assert abs(r['same_label']+r['mixed']-1)<1e-12 and abs(r['background_background']+r['liver_liver']-r['same_label'])<1e-12
 assert row['attracted']['pairs']<=row['random_unfiltered']['pairs']
 if row['random_unfiltered']['pairs']:assert abs(row['attraction_rate']-row['attracted']['pairs']/row['random_unfiltered']['pairs'])<1e-12
for typ in ['intra','knn','negative']:
 for kind in ['random_unfiltered','attracted']:
  assert p['summary'][typ+'/all'][kind]['pairs']==sum(p['summary'][typ+'/'+g][kind]['pairs'] for g in ['same_patient','cross_patient'])
assert p['summary']['knn/same_slice']['image_pair_draws']==0 and p['summary']['negative/same_slice']['image_pair_draws']==0
assert p['summary']['intra/all']['image_pair_draws']==p['summary']['knn/all']['image_pair_draws']==472*7
assert p['cache_locality']['image_edges']==472*7
model,ck=load(setup['checkpoints']['seed0']['path']);lossfn=model.contrastive_corr_loss_fn
f=torch.randn(2,768,11,11);g=torch.randn_like(f);ones=torch.ones(2,1,11,11)
with torch.inference_mode():
 original,cd=lossfn.helper(f,g,ones,ones,0);aff=-original;errors={}
 for name,shift in setup['shifts'].items():
  loss,cd=lossfn.helper(f,g,ones,ones,shift);recovered=shift-loss/cd;errors[name]=float((recovered-aff).abs().max());assert errors[name]<1e-6
zero_majority=[];runtimes={}
for name in ['seed0','seed1','seed2','best600','final']:
 r=json.loads((OUT/f'codes_{name}.json').read_text());assert r['backbone_all_parameters_equal'] and r['own_probe_assignments_equal_forensics'];assert r['train_patches']==1869*196
 assert all(e['feature_max_abs_difference']==e['code_max_abs_difference']==0 for e in r['live_forward_errors'].values())
 assert r['elapsed_seconds']<570;runtimes[name]=r['elapsed_seconds']
 for seed,k in r['kmeans'].items():
  for mode in ['nearest','matched_bilinear']:
   assert k[mode]['pixels']==472*224*224
   if k[mode]['majority']['iou']==0:zero_majority.append([name,seed,mode])
for name,digest in setup['source_sha256'].items():assert hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()==digest
assert not subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=ROOT,text=True).strip()
for path in OUT.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
r=dict(script=__file__,status='PASS',affinity_exposure_max_errors=errors,source_hashes_unchanged=True,no_tracked_source_changes=True,purity_conservation_checks='PASS',all_five_checkpoint_head_probe_reproductions='PASS',code_fit_zero_majority_cases=zero_majority,code_script_runtimes=runtimes,setup_runtime=setup['elapsed_seconds'],purity_runtime=p['elapsed_seconds'],elapsed_seconds=elapsed())
dump('verification.json',r);note('Final checks: `audit/scratch/pairs/verify.py` → `verification.json`; '+json.dumps(r));print(json.dumps(r,indent=2))
