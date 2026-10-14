"""Validate scoring/generalization and retain reproducibility evidence, no fits."""
from common import *
import subprocess
from sklearn.metrics import mutual_info_score
node=next(n for n in ast.parse((FORE/'task2.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='agreement')
exec(compile(ast.Module(body=[node],type_ignores=[]),str(FORE/'task2.py'),'exec'),globals())
rng=np.random.default_rng(SEED)
for K in [8,16]:
 pr=rng.integers(0,K,(9,11));y=rng.integers(0,2,(9,11));conf=confusion(pr,y,K)
 assert conf.sum()==pr.size and conf[:,1].sum()==y.sum()
 report=mapping_any_k(conf)
 for key in ['majority','optimal']:
  r=report[key];mask=np.isin(pr,r['clusters']);truth=y==1
  assert r['tp']==int((mask&truth).sum()) and r['fp']==int((mask&~truth).sum()) and r['fn']==int((~mask&truth).sum())
 assert report['optimal']['iou']>=report['majority']['iou']
 if K==8:assert report==mapping_report(conf)
rows=[json.loads(p.read_text()) for p in sorted(OUT.glob('k[0-9]*_c[01]_s[012].json'))];assert len(rows)==12
for r in rows:
 assert r['fit_patches']==366324 and r['val_slices']==472
 assert r['elapsed_seconds']<570
 total=np.asarray(r['confusion']);assert np.array_equal(total,sum(np.asarray(x) for x in r['patient_confusions'].values()))
 assert r['lopo_pooled']['iou']<=r['optimal']['iou']+1e-12
heads=json.loads((OUT/'heads.json').read_text());assert len(heads)==5
for r in heads.values():assert r['elapsed_seconds']<570
partitions={}
for center in [0,1]:
 for K in [8,16]:
  for a,b in [(0,1),(0,2),(1,2)]:
   x=np.load(OUT/f'k{K}_c{center}_s{a}_assignments.npy',mmap_mode='r');y=np.load(OUT/f'k{K}_c{center}_s{b}_assignments.npy',mmap_mode='r');cm=np.zeros((K,K),dtype=np.int64)
   for start in range(0,472,16):cm+=np.bincount((np.asarray(x[start:start+16],dtype=np.int64)*K+np.asarray(y[start:start+16],dtype=np.int64)).ravel(),minlength=K*K).reshape(K,K)
   partitions[f'K{K} center{center} seed{a} vs seed{b}']=agreement(cm)['ARI']
search_roots=[ROOT/'audit/scratch/diag',ROOT/'audit/run_records']
hits=[str(p) for d in search_roots for p in d.rglob('*') if p.is_file() and (p.suffix in ['.pkl','.pickle','.joblib','.npz'] or any(s in p.name.lower() for s in ['linear','logistic','probe']))]
linear={str(p):json.loads(p.read_text())['linear']['model_saved'] for p in [ROOT/'audit/scratch/diag/ceiling_b12_c0.json',ROOT/'audit/scratch/diag/ceiling_b12_c1.json']}
result=dict(script=__file__,synthetic_confusion_and_mapping_checks='PASS K8 and K16',fit_rows=len(rows),head_rows=len(heads),all_successful_fits_under_570_seconds=True,label_free_partition_ARI=partitions,linear_artifact_search_roots=[str(p) for p in search_roots],linear_artifact_filename_hits=hits,DIAG_model_saved_flags=linear,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),tracked_diff=subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=ROOT,text=True),source_reuse_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [FORE/'common.py',FORE/'task1.py',FORE/'task2.py']},elapsed_seconds=elapsed())
dump('verification.json',result);print(json.dumps(result,indent=2),flush=True)
