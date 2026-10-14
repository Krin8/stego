"""Train-image-only user heuristic, not Paper A. No labels/validation pixels/cache writes."""
from fixes3_common import *
import json,time
from PIL import Image
from omegaconf import OmegaConf
from modules import BiomedCLIPFeaturizer,ContrastiveCorrelationLoss,sample
from data import CHAOS
from chaos_preprocessing import get_chaos_transform
from knn_cache import expected_identity,cache_path,read_cache
SEED=20260930;start=time.monotonic();torch.manual_seed(SEED)
# Reject reads of any annotation or validation image, including via symlinks.
def no_labels(event,args):
    if event=='open' and not isinstance(args[0],int):
        path=str(Path(args[0]).resolve())
        if '/preprocessed_v2/' in path and ('/labels/' in path or '/val/' in path):
            raise PermissionError('Calibration may not read labels or validation data: '+path)
sys.addaudithook(no_labels)
cfg=OmegaConf.load(ROOT/'src/configs/train_config.yaml')
ds=CHAOS(str(ROOT/'src/pytorch_data_dir'),'CT','train',get_chaos_transform(),get_chaos_transform(is_label=True),2,True)
rng=np.random.default_rng(SEED);ix=[];patients=sorted({r['patient_id'] for r in ds.records},key=int)
for pi,p in enumerate(patients):
    pool=[i for i,r in enumerate(ds.records) if r['patient_id']==p]
    for stratum in np.array_split(pool,200//len(patients)+(pi<200%len(patients))):ix.append(int(rng.choice(stratum)))
identity=expected_identity(ds,cfg);cache=cache_path(ds.root,identity);nns=read_cache(cache,identity)
lookup={idx:i for i,idx in enumerate(ix)}
edges=[(i,lookup[int(j)]) for i,idx in enumerate(ix) for j in nns[idx,1:cfg.num_neighbors+1] if int(j) in lookup]
net=BiomedCLIPFeaturizer(cfg.dim,cfg).cpu().eval();features=[]
with torch.no_grad():
 for offset in range(0,200,4):
    images=[]
    for index in ix[offset:offset+4]:
        with Image.open(ds.samples[index][0]) as im:images.append(ds.transform(im))
    f,_=net(torch.stack(images));features.append(f)
f=torch.cat(features);assert f.shape==(200,768,14,14)
rng=np.random.default_rng(SEED)
pairs=dict(intra=[(i,i) for i in range(200)],knn=edges,
           random=[(i,int(rng.choice([j for j in range(200) if j!=i]))) for i in range(200)])
loss=ContrastiveCorrelationLoss(cfg);values={}
with torch.no_grad():
 for name,pp in pairs.items():
    torch.manual_seed(SEED);c1=torch.rand(len(pp),11,11,2)*2-1;c2=torch.rand(len(pp),11,11,2)*2-1;parts=[]
    for offset in range(0,len(pp),16):
        batch=pp[offset:offset+16];fa=sample(f[[i for i,j in batch]],c1[offset:offset+len(batch)])
        fb=fa if name=='intra' else sample(f[[j for i,j in batch]],c2[offset:offset+len(batch)])
        ones=torch.ones(len(batch),1,11,11);negative_affinity,cd=loss.helper(fa,fb,ones,ones,0.)
        assert torch.equal(cd,torch.ones_like(cd));parts.append((-negative_affinity).numpy().ravel())
    values[name]=np.concatenate(parts)
intra=float(np.quantile(values['intra'],.25));inter=float(np.quantile(values['knn'],.25))
sets={}
for name,tail in [('B',.10),('C',.15)]:
    neg=float(np.quantile(values['random'],1-tail))
    sets[name]=dict(pos_intra_shift=intra,pos_inter_shift=inter,neg_inter_shift=neg,
                    attraction_intra=float(np.mean(values['intra']>intra)),attraction_knn=float(np.mean(values['knn']>inter)),attraction_random=float(np.mean(values['random']>neg)))
assert np.allclose([intra,inter,sets['B']['neg_inter_shift']],[.4584,.4525,.7792],atol=5e-5,rtol=0)
assert abs(sets['B']['attraction_random']-.10)<1e-5 and abs(sets['C']['attraction_random']-.15)<1e-5
result=dict(script=__file__,seed=SEED,heuristic='User heuristic, NOT from Paper A; positive25th and random90th/85th percentiles',train_sample_ids=[ds.sample_ids[i] for i in ix],patient_count=len(patients),knn_edge_count=len(edges),baseline=dict(pos_intra_shift=float(cfg.pos_intra_shift),pos_inter_shift=float(cfg.pos_inter_shift),neg_inter_shift=float(cfg.neg_inter_shift),random_attraction=float(np.mean(values['random']>cfg.neg_inter_shift))),candidates=sets,cache=str(cache),label_or_validation_reads=0,elapsed_seconds=time.monotonic()-start)
(OUT/'calibration.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='train_sample_ids'},indent=2))
