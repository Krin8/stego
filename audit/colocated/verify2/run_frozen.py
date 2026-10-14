import os, sys, json, torch, numpy as np, time
from pathlib import Path
from sklearn.cluster import KMeans
import pydicom
from scipy import ndimage as ndi

sys.path.append("experiments/stego_colocated")
import os
os.environ['STEGO_STAGE'] = 'evaluation'
import tempfile
sys.modules['wandb'] = type('Mock', (), {})()
tempfile.tempdir = str(Path("experiments/stego_colocated").resolve())
import runtime as rt
from runtime import records, seed_all, read, load_image, frozen_backbone, image_tensor, crop_image, distances, inverse_grid, resize224
from loss import ColocatedCorrelationLoss
from model import StegoHead

ROOT = Path(".").resolve()
sys.path.append(str(ROOT / "audit/scratch/forensics"))
sys.path.append(str(ROOT / "audit/improve"))
try:
    from common import scores
    from pipeline import patient_stats
except:
    pass

def largest(mask,structure):
    lab,n=ndi.label(mask,structure=structure)
    if not n:return mask
    counts=np.bincount(lab.ravel());counts[0]=0
    return lab==counts.argmax()

def guided(image,field):
    size=17;mi=ndi.uniform_filter(image,size=size,mode='reflect');mp=ndi.uniform_filter(field,size=size,mode='reflect')
    var=ndi.uniform_filter(image*image,size=size,mode='reflect')-mi*mi
    cov=ndi.uniform_filter(image*field,size=size,mode='reflect')-mi*mp
    aa=cov/(var+.01);bb=mp-aa*mi
    return np.clip(ndi.uniform_filter(aa,size=size,mode='reflect')*image+ndi.uniform_filter(bb,size=size,mode='reflect'),0,1)

def score_fields_local(z,im,r,centers,selected,temp):
    G = 28
    d=distances(z.reshape(-1,768),centers)
    assignment=d.argmin(1).reshape(G,G).astype(np.uint8)
    native=inverse_grid(assignment,r,outside=8).astype(np.uint8)
    result={}
    weights=np.exp(-(d-d.min(1,keepdims=True))/temp);weights/=weights.sum(1,keepdims=True)
    for rule,cluster in selected.items():
        hard=(native==cluster) if cluster is not None else np.zeros(tuple(r['shape']),bool)
        field=weights[:,cluster].reshape(G,G) if cluster is not None else np.zeros((G,G),np.float32)
        soft=inverse_grid(field,r,mode='bilinear',outside=0)
        pred=largest(guided(im,soft)>=.5,np.ones((3,3),int))
        pred=ndi.binary_fill_holes(pred)
        yy,xx=np.mgrid[-3:4,-3:4]
        pred=ndi.binary_closing(np.pad(pred,3),structure=xx*xx+yy*yy<=9)[3:-3,3:-3]
        result[rule]=(hard,pred)
    return assignment,native,result

OUT = ROOT / "audit/colocated/verify2"
OUT.mkdir(parents=True, exist_ok=True)
rr = records()
train_ids = [i for i, r in enumerate(rr) if r['split'] == 'train']
val_ids = [i for i, r in enumerate(rr) if r['split'] == 'val']

# Figure out the path to B2_448_features.npy
for p in [ROOT/"archive/features/B2_448_features.npy", ROOT/"audit/preflight/B2_448_features.npy", ROOT/"audit/improve2/B2_448_features.npy", ROOT/"audit/improve/B2_448_features.npy", ROOT.parent/"STEGO/experiments/stego_colocated/B2_448_features.npy", ROOT/"audit/scratch/features/B2_448_features.npy"]:
    if p.exists():
        features = np.load(p, mmap_mode='r')
        break
else:
    # Need to find the OLD path from runtime.py
    import runtime as rt
    features = np.load(rt.OLD / "B2_448_features.npy", mmap_mode='r')

texts = np.load(rt.BASE / "text_embeddings.npy")
net = frozen_backbone()
full = net.full_model

all_results = {}
val_y = {224: np.load(rt.OLD / 'scoring_masks_224.npy', mmap_mode='r'), 512: np.load(rt.OLD / 'scoring_masks_native.npy', mmap_mode='r')}

for seed in [0, 1, 2]:
    print(f"Processing seed {seed}...")
    seed_all(seed)
    model = StegoHead().eval().requires_grad_(False)
    
    def embedding(i):
        f = torch.from_numpy(np.array(features[i])).reshape(28,28,768).permute(2,0,1)[None]
        with torch.no_grad():
            z = f[0].permute(1,2,0).contiguous().numpy()
        return z
        
    print("  Fitting K-Means...")
    sample = np.zeros((len(train_ids)*784, 768), dtype=np.float32)
    for j, i in enumerate(train_ids):
        sample[j*784:(j+1)*784] = embedding(i).reshape(784, 768)
        
    km = KMeans(n_clusters=8, random_state=seed, n_init=3, max_iter=100, tol=1e-4, algorithm='lloyd').fit(sample)
    centers = km.cluster_centers_
    d = distances(sample, centers)
    two = np.partition(d, 1, axis=1)[:, :2]
    temp = max(float(np.median(two[:,1]-two[:,0])/np.log(4)), 1e-6)
    
    print("  Naming...")
    sums = np.zeros((8,4), np.float64)
    nearest = [[] for _ in range(8)]
    fallback_count = 0
    yy,xx = np.mgrid[-5:6,-5:6]; disk = xx*xx+yy*yy<=25
    G = 28
    
    for n, i in enumerate(train_ids):
        r = rr[i]
        ds = pydicom.dcmread(r['dicom'])
        hu = ds.pixel_array.astype(np.float32)*float(getattr(ds,'RescaleSlope',1))+float(getattr(ds,'RescaleIntercept',0))
        body = ndi.binary_fill_holes(largest(ndi.binary_opening(hu>-500,structure=disk),np.ones((3,3),int)))
        ys,xs = np.where(body)
        y0,x0,side = r['box']
        gy,gx = np.mgrid[:G,:G]; cy=y0+(gy.ravel()+.5)*side/G-.5; cx=x0+(gx.ravel()+.5)*side/G-.5
        iy=np.floor(cy+.5).astype(int); ix=np.floor(cx+.5).astype(int)
        eligible=(iy>=0)&(ix>=0)&(iy<512)&(ix<512); valid=np.flatnonzero(eligible); eligible[valid]=body[iy[valid],ix[valid]]; use=np.flatnonzero(eligible)
        ya=np.clip(np.ceil(y0+gy.ravel()*side/G-.5).astype(int),0,512); yb=np.clip(np.ceil(y0+(gy.ravel()+1)*side/G-.5).astype(int),0,512)
        xa=np.clip(np.ceil(x0+gx.ravel()*side/G-.5).astype(int),0,512); xb=np.clip(np.ceil(x0+(gx.ravel()+1)*side/G-.5).astype(int),0,512)
        def rectsum(a):
            integral=np.pad(a,((1,0),(1,0))).cumsum(0,dtype=np.float64).cumsum(1,dtype=np.float64); return integral[yb,xb]-integral[ya,xb]-integral[yb,xa]+integral[ya,xa]
        count=rectsum(body); total=rectsum(hu*body)
        means=np.divide(total,count,out=np.zeros_like(total),where=count>0); empty=use[count[use]==0]
        means[empty]=hu[iy[empty],ix[empty]]; fallback_count+=len(empty)
        
        z = embedding(i).reshape(-1,768); d_dist = distances(z, centers); assignment = d_dist.argmin(1)
        nx = (cx-xs.min())/max(1,xs.max()-xs.min()); ny = (cy-ys.min())/max(1,ys.max()-ys.min())
        for cluster in range(8):
            u = use[assignment[use]==cluster]
            sums[cluster,0] += len(u)
            sums[cluster,1:] += [nx[u].sum(), ny[u].sum(), means[u].sum()]
            v = d_dist[use,cluster]
            cut = np.partition(v, min(199,len(v)-1))[min(199,len(v)-1)]
            pool = use[v<=cut]
            order = pool[np.lexsort((pool, d_dist[pool,cluster]))][:200]
            candidates = [(float(d_dist[t,cluster]), i, int(t), float(cx[t]), float(cy[t])) for t in order]
            nearest[cluster] = sorted(nearest[cluster]+candidates)[:200]
            
    stats = []
    for k in range(8):
        count = sums[k,0]
        x,y,h = sums[k,1:]/count if count else [float(1),float(1),float(0)]
        stats.append(dict(cluster=k, body_patches=int(count), area=float(count/max(1,sums[:,0].sum())), centroid_x=float(x), centroid_y=float(y), mean_HU=float(h), candidate=bool(count and x<.5 and y<.6 and 40<=h<=180)))
    possible = [x for x in stats if x['candidate']]
    n2 = max(possible, key=lambda x:x['area'])['cluster'] if possible else None
    
    keys=sorted(set((i,t,x,y) for nn in nearest for _,i,t,x,y in nn))
    emb=np.empty((len(keys),512),np.float32)
    for s in range(0,len(keys),16):
        inp=[]
        for i,t,x,y in keys[s:s+16]:
            inp.append(image_tensor(crop_image(load_image(rr[i]),[int(np.floor(y+.5))-56,int(np.floor(x+.5))-56,112]),224))
        with torch.no_grad():
            emb[s:s+len(inp)]=full.encode_image(torch.stack(inp),normalize=True).numpy()
            
    lookup={k:i for i,k in enumerate(keys)}
    avg=np.stack([emb[[lookup[(i,t,x,y)] for _,i,t,x,y in nn]].mean(0) for nn in nearest])
    avg/=np.maximum(np.linalg.norm(avg,axis=1,keepdims=True),1e-12)
    sim=avg@texts.T
    std=sim.std(0,keepdims=True)
    zz=np.divide(sim-sim.mean(0,keepdims=True),std,out=np.zeros_like(sim),where=std>0)
    n1=int(zz[:,0].argmax())
    selected=dict(N1=n1, N2=n2, N3=n1 if n1==n2 else None)
    
    print("  Predicting & Scoring...")
    run_res = {"selected": selected, "rules": {}}
    
    for rule in ["N1", "N2", "N3"]:
        run_res["rules"][rule] = {"hard": {}, "F1": {}}
        for res in [224, 512]:
            run_res["rules"][rule]["hard"][str(res)] = {"patients": {}}
            run_res["rules"][rule]["F1"][str(res)] = {"patients": {}}
            
    for j, i in enumerate(val_ids):
        z = embedding(i)
        im = load_image(rr[i])
        assignment, native_pred, res_masks = score_fields_local(z, im, rr[i], centers, selected, temp)
        
        for rule in ["N1", "N2", "N3"]:
            for kind_idx, kind in enumerate(["hard", "F1"]):
                pred512 = res_masks[rule][kind_idx]
                pred224 = resize224(pred512)
                
                for res, pred in [(224, pred224), (512, pred512)]:
                    co = np.bincount((np.asarray(pred,np.int64)*2+np.asarray(val_y[res][i],np.int64)).ravel(), minlength=4).reshape(2,2)
                    p_id = str(rr[i]['patient'])
                    
                    if p_id not in run_res["rules"][rule][kind][str(res)]["patients"]:
                        run_res["rules"][rule][kind][str(res)]["patients"][p_id] = {"tp": 0, "fp": 0, "fn": 0}
                        
                    run_res["rules"][rule][kind][str(res)]["patients"][p_id]["tp"] += int(co[1,1])
                    run_res["rules"][rule][kind][str(res)]["patients"][p_id]["fp"] += int(co[1,0])
                    run_res["rules"][rule][kind][str(res)]["patients"][p_id]["fn"] += int(co[0,1])
                    
    all_results[str(seed)] = run_res

print("===JSON_OUTPUT_START===")
print(json.dumps({"results": all_results}, indent=2))
print("===JSON_OUTPUT_END===")
print("Untrained evaluation complete.")
