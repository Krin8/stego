"""Read-only checkpoint/code comparison; tiny frozen-input CPU checks only."""
import ast,csv,hashlib,importlib.util,json,os,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
import numpy as np,torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(2);torch.set_grad_enabled(False)
P=ROOT/'experiments/stego_colocated';sys.path.insert(0,str(P))
from model import StegoHead
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
# Extract only the optimized class AST: do not execute its refitting/scoring script.
path=ROOT/'audit/ensemble/verify4/run_p1_optimized.py';tree=ast.parse(path.read_text())
node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='StegoHead')
ns={'nn':nn};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
features=np.load(ROOT/'audit/improve/B2_448_features.npy',mmap_mode='r',allow_pickle=False)
records=read(ROOT/'audit/naming/image_records.json')
ids=[0]+[next(i for i,r in enumerate(records) if r['split']=='val' and r['patient']==p) for p in ['5','6','8','30']]
x=torch.from_numpy(np.stack([features[i] for i in ids])).reshape(-1,28,28,768).permute(0,3,1,2)
checkpoints=[]
for seed in [0,1,2]:
    ck=P/f'P1/stage2/checkpoints/seed{seed}/final.pt';state=torch.load(ck,map_location='cpu',weights_only=True,mmap=True)
    full=StegoHead().eval().requires_grad_(False);full.load_state_dict(state['state_dict'],strict=True)
    reduced=ns['StegoHead']().eval().requires_grad_(False);result=reduced.load_state_dict(state['state_dict'],strict=False)
    a=full.embed(x);b=reduced.embed(x);bn=F.normalize(b,dim=1)
    current_hashes={name:sha(P/name)==digest for name,digest in state.get('source_sha256',{}).items() if (P/name).exists()}
    ckinfo=dict(path=str(ck.relative_to(ROOT)),bytes=ck.stat().st_size,sha256=sha(ck),
        step=state.get('step'),seed=state.get('seed'),state_keys=list(state['state_dict']),
        config_sha_matches=state.get('config_sha256')==sha(P/'P1/stage2/configs/locked.json'),
        current_source_hash_matches=current_hashes,
        optimizer_group_parameter_counts=[len(g['params']) for g in state['optimizer']['param_groups']],
        optimizer_learning_rates=[g['lr'] for g in state['optimizer']['param_groups']],
        optimized_missing_keys=result.missing_keys,optimized_unexpected_keys=result.unexpected_keys,
        feature_shape=list(x.shape),feature_norm_mean=float(x.norm(dim=1).mean()),
        full_output_norm_mean=float(a.norm(dim=1).mean()),optimized_output_norm_mean=float(b.norm(dim=1).mean()),
        mean_direction_cosine=float((a*bn).sum(1).mean()),max_abs_output_difference=float((a-b).abs().max()),
        deterministic=bool(torch.equal(full.embed(x),a)))
    checkpoints.append(ckinfo)
results=read(P/'P1/stage2/metrics/results.json');scores=[]
for seed in [0,1,2]:
    r=results['results'][str(seed)]
    for resolution in ['224','512']:
        p=r['rules']['N3']['F1'][resolution];s=p['summary']['pooled']
        tp=sum(v['pooled']['tp'] for v in p['patients'].values());fp=sum(v['pooled']['fp'] for v in p['patients'].values());fn=sum(v['pooled']['fn'] for v in p['patients'].values())
        scores.append(dict(seed=seed,scale=resolution,names=r['selected'],patient_macro_iou=100*s['iou']['mean'],patient_sd_iou=100*s['iou']['std'],patient_macro_dice=100*s['dice']['mean'],all_patient_pooled_iou=100*tp/(tp+fp+fn),all_patient_pooled_dice=200*tp/(2*tp+fp+fn)))
matches=[]
with (OUT/'FILES.csv').open() as f:
    rows=list(csv.DictReader(f))
for r in rows:
    if r['role'] not in ['report/document','log/text','config/metadata/results','project-python']:continue
    p=ROOT/r['path']
    if p.suffix.lower() not in ['.py','.md','.txt','.log','.json','.jsonl','.csv','.yaml','.yml']:continue
    if int(r['bytes'])>40_000_000:continue
    try:text=p.read_text()
    except (UnicodeError,OSError):continue
    for m in list(re.finditer(r'(?<![\d.])(?:83\.92|86\.72)(?!\d)',text))[:10]:
        matches.append(dict(path=r['path'],line=text.count('\n',0,m.start())+1,context=text[max(0,m.start()-90):m.end()+110].replace('\n',' ')))
output=dict(producer=__file__,labels_opened=False,training_steps=0,cache_rows_checked=ids,
    feature_cache_shape=list(features.shape),checkpoints=checkpoints,saved_original_F1_scores=scores,claimed_score_text_matches=matches)
(OUT/'artifact_checks.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
