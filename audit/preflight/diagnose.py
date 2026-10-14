"""Read-only, label-free preflight; all writes restricted to this directory."""
import os, sys, json, ast, hashlib, time, tempfile, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT/'experiments/stego_biomedclip'
sys.dont_write_bytecode = True
os.environ.setdefault('HF_HOME', str(Path.home()/'.cache/huggingface'))
os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
 OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
 MPLBACKEND='Agg', MPLCONFIGDIR=str(HERE/'mpl'), XDG_CACHE_HOME=str(HERE/'.cache'))
tempfile.tempdir = str(HERE)
READS = set()
def guard(event, args):
    if event == 'open' and not isinstance(args[0], int):
        p = Path(os.fsdecode(args[0])).resolve(); mode, flags = args[1:]
        write = (isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if write and not p.is_relative_to(HERE): raise PermissionError('Write outside preflight: '+str(p))
        if not write:
            s = str(p).lower()
            forbidden = any(t in s for t in ['/ground/', '/labels/', '/masks/', 'scoring_masks', '/preprocessed_v2/val/'])
            forbidden |= p.name.endswith(('_scores.json','_results.json','_labels.npy'))
            forbidden |= p.parent == ROOT/'audit/final_pipeline' and p.suffix == '.json' and not p.name.endswith('_fit.json')
            if forbidden: raise PermissionError('Labels/validation excluded: '+str(p))
            READS.add(str(p))
    elif event == 'socket.connect': raise PermissionError('Offline only')
    elif event in ['os.remove','os.rmdir','os.mkdir','os.rename']:
        for v in args[:2] if event == 'os.rename' else args[:1]:
            p = Path(os.fsdecode(v)).resolve()
            if not p.is_relative_to(HERE) and not (event == 'os.mkdir' and p.exists()):
                raise PermissionError('Mutation outside preflight: '+str(p))
sys.addaudithook(guard)
import numpy as np
import torch
import torch.nn.functional as F
from types import SimpleNamespace
torch.set_num_threads(2)
torch.set_grad_enabled(False)
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(ROOT/'src'))
from reference_loss import ContrastiveCorrelationLoss
from model import StegoHead, objective

def dump(name, value): (HERE/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

CFG = read(EXP/'configs/locked.json')
assert sha(EXP/'configs/locked.json') == (EXP/'configs/locked.sha256').read_text().strip()
RR = read(ROOT/'audit/naming/image_records.json')  # Image-only metadata, no mask paths.
TRAIN = np.array([i for i,r in enumerate(RR) if r['split']=='train'])
assert len(TRAIN)==1869
SEED=20261003
SAMPLE=np.random.default_rng(SEED).choice(len(TRAIN),200,replace=False)
GLOBAL=TRAIN[SAMPLE]
FEATURES=np.load(ROOT/'audit/improve/B2_448_features.npy', mmap_mode='r', allow_pickle=False)
NEIGHBORS=np.load(EXP/'logs/train_neighbors.npy',allow_pickle=False)
assert NEIGHBORS.shape==(1869,30) and NEIGHBORS.min()>=0 and NEIGHBORS.max()<1869
assert np.all(NEIGHBORS != np.arange(1869)[:,None])

def batch(local):
    return torch.from_numpy(np.stack([FEATURES[TRAIN[int(i)]] for i in local])).reshape(-1,28,28,768).permute(0,3,1,2).contiguous()

def monitors(code):
    z=F.normalize(code,dim=1)
    means=z.flatten(2).mean(2); cos=means@means.T
    return dict(code_std=float(z.flatten(2).std(2).mean()),
      cross_image_cosine=float(cos[~torch.eye(len(means),dtype=torch.bool)].mean()))

def diversity(x):
    # x: images x 784 x 768, each patch L2 normalized. Full covariance, no PCA truncation.
    t=torch.from_numpy(np.asarray(x).copy()).double()
    means=t.mean(1); cosine=means@means.T
    normalized=F.normalize(means,dim=1); gapcos=normalized@normalized.T
    off=~torch.eye(len(t),dtype=torch.bool)
    flat=t.reshape(-1,768); mean=flat.mean(0); centered=flat-mean
    cov=centered.T@centered/(len(flat)-1)
    eig=torch.linalg.eigvalsh(cov).clamp_min(0).flip(0)
    p=eig/eig.sum(); nonzero=p[p>0]
    std=flat.std(0)
    return dict(images=len(t),patches=len(flat),patch_norm_min=float(t.norm(dim=2).min()),
      patch_norm_max=float(t.norm(dim=2).max()),patch_pair_cosine=float(cosine[off].mean()),
      normalized_gap_cosine=float(gapcos[off].mean()),channel_std_mean=float(std.mean()),
      channel_std_min=float(std.min()),channel_std_max=float(std.max()),channel_std=std.tolist(),
      explained_variance_ratio=p.tolist(),pc1=float(p[0]),pc10=float(p[:10].sum()),
      pc50=float(p[:50].sum()),effective_rank=float(torch.exp(-(nonzero*nonzero.log()).sum())))

def core():
    dump('sample.json',dict(seed=SEED,train_population=len(TRAIN),sample_count=len(SAMPLE),
      local_indices=SAMPLE.tolist(),global_indices=GLOBAL.tolist(),sample_ids=[RR[i]['sample_id'] for i in GLOBAL],
      patient_counts={p:sum(RR[i]['patient']==p for i in GLOBAL) for p in sorted({RR[i]['patient'] for i in GLOBAL},key=int)}))
    centers=torch.from_numpy(np.load(ROOT/'audit/final_pipeline/inductive_seed0_centers.npy',allow_pickle=False))
    assert centers.shape==(8,768)
    rng=np.random.default_rng(SEED+1)
    positives=np.array([NEIGHBORS[j,rng.integers(CFG['num_neighbors'])] for j in SAMPLE])
    dump('pair_plan.json',dict(positive_local_indices=positives.tolist(),positive_global_indices=TRAIN[positives].tolist(),
      top_neighbors_used=7,stored_k=30,spatial_samples=11,negative_permutations=5,batch_size=16,
      centers_path='audit/final_pipeline/inductive_seed0_centers.npy',centers_sha256=sha(ROOT/'audit/final_pipeline/inductive_seed0_centers.npy')))
    shifts={'Pinned':[.18,.12,.46], 'C':[.4584466218948364,.45247218012809753,.7459585666656494],
      'D':[.7965070605278015,.7820567488670349,.7791634202003479],
      'E':[.8634735345840454,.8470812439918518,.8234664797782898]}
    affinities=[[],[],[]]
    sums={g:{kind:np.zeros(4) for kind in ['constant','random','cluster']} for g in shifts}
    # Capture affinity through the actual helper: with constant codes and shift zero, -loss == fd.
    class Capture(ContrastiveCorrelationLoss):
        def helper(self,f1,f2,c1,c2,shift):
            const1=torch.ones_like(c1[:,:1]);const2=torch.ones_like(c2[:,:1])
            a,_=super().helper(f1,f2,const1,const2,0.)
            index=min(self.calls,2);self.calls+=1
            affinities[index].append((-a).numpy().ravel().copy())
            return super().helper(f1,f2,c1,c2,shift)
    for start in range(0,200,16):
        ids=SAMPLE[start:start+16];pos=positives[start:start+16]
        f,fp=batch(ids),batch(pos)
        torch.manual_seed(SEED+start+10)
        def codes(x,kind):
            if kind=='constant': return torch.ones(len(x),70,28,28)
            if kind=='random': return F.normalize(torch.randn(len(x),70,28,28),dim=1)
            flat=x.permute(0,2,3,1).reshape(-1,768)
            d=flat.square().sum(1)[:,None]+centers.square().sum(1)[None]-2*flat@centers.T
            k=d.argmin(1).reshape(len(x),28,28)
            return F.one_hot(k,num_classes=70).permute(0,3,1,2).float()
        codepairs={kind:(codes(f,kind),codes(fp,kind)) for kind in ['constant','random','cluster']}
        for group,ss in shifts.items():
            c=SimpleNamespace(**{**CFG,**dict(zip(['pos_intra_shift','pos_inter_shift','neg_inter_shift'],ss))})
            for kind,(code,cp) in codepairs.items():
                crit=Capture(c) if group=='Pinned' and kind=='constant' else ContrastiveCorrelationLoss(c)
                crit.calls=0
                torch.manual_seed(SEED+start)
                total,parts=objective(crit,c,f,fp,code,cp)
                sums[group][kind]+=len(ids)*np.array([float(parts[k]) for k in ['pos_intra','pos_inter','neg_inter']]+[float(total)])
        print('loss sample',start+len(ids),flush=True)
    aa=[np.concatenate(v) for v in affinities]
    result=dict(seed=SEED,sample_count=200,weights=[CFG[k] for k in ['pos_intra_weight','pos_inter_weight','neg_inter_weight']],
      feature_dropout='off (normalized frozen tokens)',affinity_count=[len(a) for a in aa],
      affinity_mean=[float(a.mean()) for a in aa],quantiles={str(q):[float(np.quantile(a,q)) for a in aa] for q in [.05,.1,.2,.25,.5,.75,.8,.85,.9,.95,.99]},groups={})
    for group,ss in shifts.items():
        result['groups'][group]=dict(shifts=ss,attraction=[float((a>s).mean()) for a,s in zip(aa,ss)],
          losses={kind:dict(zip(['pos_intra','pos_inter','neg_inter','total'],(v/200).tolist())) for kind,v in sums[group].items()})
    dump('objective.json',result)
    del aa,affinities
    print('diversity final',flush=True)
    dump('diversity_final.json',diversity(np.stack([FEATURES[i] for i in GLOBAL])))
    monitor_rng=np.random.default_rng(CFG['monitor_seed'])
    patients=sorted({RR[i]['patient'] for i in TRAIN},key=int)
    mid=[int(monitor_rng.choice([j for j,i in enumerate(TRAIN) if RR[i]['patient']==p])) for p in patients]
    fm=batch(mid);initial=[]
    for seed in [0,1,2]:
        torch.manual_seed(seed);head=StegoHead().eval()
        initial.append(dict(seed=seed,**monitors(head(fm)[1])))
    dump('initialization.json',dict(monitor_global_indices=TRAIN[mid].tolist(),rows=initial,
      reset_parameters_source=torch.nn.Conv2d.reset_parameters.__code__.co_filename,
      reset_parameters_line=torch.nn.Conv2d.reset_parameters.__code__.co_firstlineno))
    decoder=json.JSONDecoder(); trajectories={}
    for seed in [0,1]:
        path=EXP/f'logs/seed{seed}_train.jsonl';raw=path.read_text();rows=[];offset=0;records=0;backward=[];last=-1
        while offset<len(raw):
            i=raw.find('{',offset)
            if i<0:break
            try:r,end=decoder.raw_decode(raw,i)
            except json.JSONDecodeError: offset=i+1;continue
            offset=end;records+=1
            if r.get('batch',last)<last:backward.append(dict(record=records,from_batch=last,to_batch=r['batch']))
            last=r.get('batch',last)
            if 'code_std' in r:rows.append(dict(record_ordinal=records,byte_offset=i,**r))
        trajectories[str(seed)]=dict(source=str(path.relative_to(ROOT)),source_sha256=sha(path),records=records,
          backward_batch_transitions=backward,monitor_rows=rows)
    dump('trajectories.json',trajectories)
    print('core complete',flush=True)

def extract():
    from PIL import Image
    from torchvision.transforms import functional as TF, InterpolationMode
    from scipy import ndimage as ndi
    from chaos_preprocessing import MEAN,STD
    ns=dict(np=np,torch=torch,TF=TF,InterpolationMode=InterpolationMode,MEAN=MEAN,STD=STD,ndi=ndi)
    path=ROOT/'audit/final_pipeline/segment_ct.py'
    tree=ast.parse(path.read_text())
    for name in ['body_box','crop_image','image_tensor']:
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    from modules import BiomedCLIPFeaturizer
    net=BiomedCLIPFeaturizer(70,SimpleNamespace(projection_type=None,dropout=False,use_text_prompts=False)).cpu().eval().requires_grad_(False)
    original=net.model.get_intermediate_layers
    captured=[]
    def capture(*args,**kwargs):
        layers=original(args[0],n=[8,10,11],reshape=True,norm=True)
        captured[:]=[F.normalize(t,dim=1) for t in layers]
        return (layers[-1],)
    net.model.get_intermediate_layers=capture
    arrays={b:np.lib.format.open_memmap(HERE/f'block{b}_features.npy',mode='w+',dtype='float32',shape=(200,784,768)) for b in [9,11]}
    start=time.monotonic();errors=[];boxes_agree=True
    with torch.inference_mode():
        for offset in range(0,200,4):
            rows=GLOBAL[offset:offset+4];images=[]
            for i in rows:
                r=RR[i]; assert r['split']=='train'
                with Image.open(r['image']) as im:a=np.array(im,np.float32)/65535.
                box,_=ns['body_box'](a);boxes_agree &= box==r['box']
                images.append(ns['image_tensor'](ns['crop_image'](a,box),448))
            f,_=net(torch.stack(images));f=F.normalize(f,dim=1)
            assert tuple(f.shape[1:])==(768,28,28)
            cached=torch.from_numpy(np.stack([FEATURES[i] for i in rows])).reshape(-1,28,28,768).permute(0,3,1,2)
            errors.append(float((f-cached).abs().max()))
            for b,t in zip([9,11],captured):arrays[b][offset:offset+len(rows)]=t.permute(0,2,3,1).reshape(-1,784,768).numpy()
            if offset==0:
                first=f.clone();again=F.normalize(net(torch.stack(images))[0],dim=1)
                deterministic_error=float((first-again).abs().max())
            print('extract',offset+len(rows),'seconds',round(time.monotonic()-start,1),'cache maxdiff',max(errors),flush=True)
    for a in arrays.values():a.flush()
    dump('extract_checks.json',dict(images=200,threads=torch.get_num_threads(),device='cpu',seconds=time.monotonic()-start,
      trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),eval=not net.training,
      deterministic_max_error=deterministic_error,final_cached_max_error=max(errors),body_boxes_agree=bool(boxes_agree),
      layers_one_based=[9,11,12],normalization='shared final LayerNorm, CLS excluded, then patch L2',
      backbone_sha256=sha(ROOT/'src/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_pytorch_model.bin')))
    del net,arrays
    for b in [9,11]:
        print('diversity',b,flush=True)
        dump(f'diversity_block{b}.json',diversity(np.load(HERE/f'block{b}_features.npy',mmap_mode='r')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['core','extract']);args=p.parse_args()
    try:
        (core if args.phase=='core' else extract)()
    finally:
        dump(args.phase+'_access.json',dict(threads=torch.get_num_threads(),labels_allowed=False,opened_read_paths=sorted(READS),script_sha256=sha(__file__)))
