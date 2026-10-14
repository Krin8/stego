import sys
import os
import json
import numpy as np
import pydicom
from pathlib import Path
from scipy import ndimage as ndi
from sklearn.cluster import KMeans

# Set working directory correctly if running from IDE
ROOT = Path('/Users/navneetbavineni/STEGO').resolve()
sys.path.insert(0, str(ROOT / 'audit' / 'final_pipeline_n3'))
sys.path.insert(0, str(ROOT / 'experiments' / 'stego_colocated'))

import runtime as rt
rt.PACKAGE = ROOT
rt.STAGE = 'evaluation'

from runtime import records, seed_all, load_image, frozen_backbone, image_tensor, crop_image, distances, inverse_grid, resize224
from segment_ct import guided, largest

OUT = ROOT / "audit/ensemble/verify4"
rr = records()
train_ids = [i for i, r in enumerate(rr) if r['split'] == 'train']
val_ids = [i for i, r in enumerate(rr) if r['split'] == 'val']

features = np.load(ROOT / "audit/improve/B2_448_features.npy", mmap_mode='r')
texts = np.load(rt.BASE / "text_embeddings.npy")
net = frozen_backbone()
full = net.full_model
val_y = {512: np.load(rt.OLD / 'scoring_masks_native.npy', mmap_mode='r')}

def embedding(i):
    f = torch.from_numpy(np.array(features[i])).reshape(28,28,768).permute(2,0,1)[None]
    with torch.no_grad():
        z = f[0].permute(1,2,0).contiguous().numpy()
    return z

import torch
import torch.nn as nn
from torchvision.transforms import functional as TF

class StegoHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(768, 768, 1),
            nn.ReLU(),
            nn.Conv2d(768, 70, 1)
        )
    def embed(self, x):
        return self.head(x)

def get_val_embeddings(arm, s):
    if arm == "Frozen Baseline":
        return [embedding(i).reshape(28*28, 768) for i in val_ids]
    
    seed_all(s)
    model = StegoHead().eval().requires_grad_(False)
    train_feats = np.stack([embedding(i).reshape(28*28, 768) for i in val_ids])
    f = torch.from_numpy(train_feats).reshape(len(train_feats), 28, 28, 768).permute(0,3,1,2)
    out = []
    for chunk in torch.split(f, 32):
        with torch.no_grad():
            out.append(model.embed(chunk).permute(0,2,3,1).reshape(len(chunk), 784, 70).numpy())
    return np.concatenate(out, axis=0)

train_features = np.stack([embedding(i) for i in train_ids])
outputs = {}

def process_arm(arm, seeds):
    outputs[arm] = {s: {} for s in seeds}
    for s in seeds:
        outputs[arm][s] = {'maps': {}}
        if arm == "Frozen Baseline":
            z_train = train_features.reshape(len(train_ids)*784, 768)
        else:
            seed_all(s)
            model = StegoHead().eval().requires_grad_(False)
            f = torch.from_numpy(train_features).reshape(len(train_ids), 28, 28, 768).permute(0,3,1,2)
            out = []
            for chunk in torch.split(f, 32):
                with torch.no_grad():
                    out.append(model.embed(chunk).permute(0,2,3,1).reshape(len(chunk), 784, 70).numpy())
            z_train = np.concatenate(out, axis=0).reshape(len(train_ids)*784, 70)
            
        km = KMeans(n_clusters=8, random_state=s, n_init=3, max_iter=100, tol=1e-4, algorithm='lloyd').fit(z_train)
        centers = km.cluster_centers_
        
        d = distances(z_train, centers)
        two = np.partition(d, 1, axis=1)[:, :2]
        temp = max(float(np.median(two[:,1]-two[:,0])/np.log(4)), 1e-6)
        
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
            
            z = z_train[n*784:(n+1)*784]; d_dist = distances(z, centers); assignment = d_dist.argmin(1)
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
        for sub in range(0,len(keys),16):
            inp=[]
            for i,t,x,y in keys[sub:sub+16]:
                inp.append(image_tensor(crop_image(load_image(rr[i]),[int(np.floor(y+.5))-56,int(np.floor(x+.5))-56,112]),224))
            with torch.no_grad():
                emb[sub:sub+len(inp)]=full.encode_image(torch.stack(inp),normalize=True).numpy()
                
        lookup={k:i for i,k in enumerate(keys)}
        avg=np.stack([emb[[lookup[(i,t,x,y)] for _,i,t,x,y in nn]].mean(0) for nn in nearest])
        avg/=np.maximum(np.linalg.norm(avg,axis=1,keepdims=True),1e-12)
        sim=avg@texts.T
        std=sim.std(0,keepdims=True)
        zz=np.divide(sim-sim.mean(0,keepdims=True),std,out=np.zeros_like(sim),where=std>0)
        n1=int(zz[:,0].argmax())
        n3 = n1 if n1==n2 else None
        
        val_embeds = get_val_embeddings(arm, s)
        for i_idx, i in enumerate(val_ids):
            z = val_embeds[i_idx]
            im = load_image(rr[i])
            d_val = distances(z, centers)
            
            assignment = d_val.argmin(1).reshape(G,G).astype(np.uint8)
            hard_native = (inverse_grid(assignment, rr[i], outside=8).astype(np.uint8) == n3)
            
            weights = np.exp(-(d_val-d_val.min(1,keepdims=True))/temp)
            weights /= weights.sum(1,keepdims=True)
            field = weights[:,n3].reshape(G,G) if n3 is not None else np.zeros((G,G),np.float32)
            
            outputs[arm][s]['maps'][i] = {
                'im': im,
                'soft': field,
                'hard': hard_native,
                'r': rr[i]
            }

process_arm("Frozen Baseline", [0])
process_arm("Untrained Head", [9])

def apply_postprocessing(field, im, r, g_size=25, threshold=0.30, closing_rad=3):
    soft = inverse_grid(field, r, mode='bilinear', outside=0)
    mi = ndi.uniform_filter(im, size=g_size, mode='reflect')
    mp = ndi.uniform_filter(soft, size=g_size, mode='reflect')
    var = ndi.uniform_filter(im*im, size=g_size, mode='reflect') - mi*mi
    cov = ndi.uniform_filter(im*soft, size=g_size, mode='reflect') - mi*mp
    aa = cov / (var + 0.01)
    bb = mp - aa*mi
    filt = np.clip(ndi.uniform_filter(aa, size=g_size, mode='reflect')*im + ndi.uniform_filter(bb, size=g_size, mode='reflect'), 0, 1)
    
    pred = largest(filt >= threshold, np.ones((3,3), int))
    pred = ndi.binary_fill_holes(pred)
    yy,xx = np.mgrid[-closing_rad:closing_rad+1, -closing_rad:closing_rad+1]
    pred = ndi.binary_closing(np.pad(pred,closing_rad), structure=xx*xx+yy*yy<=closing_rad**2)[closing_rad:-closing_rad, closing_rad:-closing_rad]
    return pred

def evaluate_patient(arm, seeds_to_ensemble):
    def init_stats(): return {s: {'tp': 0, 'fp': 0, 'fn': 0} for s in seeds_to_ensemble}
    patients = {}
    for i in val_ids:
        pid = rr[i]['patient']
        if pid not in patients: patients[pid] = []
        patients[pid].append(i)
        
    p_stats_a = {pid: init_stats() for pid in patients}
    p_stats_b = {pid: init_stats() for pid in patients}
    
    for pid, p_ids in patients.items():
        for i in p_ids:
            gt = (val_y[512][i] == 1)
            im = outputs[arm][seeds_to_ensemble[0]]['maps'][i]['im']
            r = rr[i]
            for s in seeds_to_ensemble:
                soft = outputs[arm][s]['maps'][i]['soft']
                hard = outputs[arm][s]['maps'][i]['hard']
                
                pred_a = apply_postprocessing(soft, im, r)
                p_stats_a[pid][s]['tp'] += np.logical_and(pred_a, gt).sum()
                p_stats_a[pid][s]['fp'] += pred_a.sum() - np.logical_and(pred_a, gt).sum()
                p_stats_a[pid][s]['fn'] += gt.sum() - np.logical_and(pred_a, gt).sum()
                
                pred_b = apply_postprocessing(hard.astype(np.float32), im, r)
                p_stats_b[pid][s]['tp'] += np.logical_and(pred_b, gt).sum()
                p_stats_b[pid][s]['fp'] += pred_b.sum() - np.logical_and(pred_b, gt).sum()
                p_stats_b[pid][s]['fn'] += gt.sum() - np.logical_and(pred_b, gt).sum()
                
    res_a = {s: {'iou': [], 'dice': []} for s in seeds_to_ensemble}
    res_b = {s: {'iou': [], 'dice': []} for s in seeds_to_ensemble}
    
    for pid in patients:
        for s in seeds_to_ensemble:
            tp, fp, fn = p_stats_a[pid][s]['tp'], p_stats_a[pid][s]['fp'], p_stats_a[pid][s]['fn']
            res_a[s]['iou'].append(tp / max(1, tp+fp+fn) * 100)
            res_a[s]['dice'].append(2*tp / max(1, 2*tp+fp+fn) * 100)
            
            tp, fp, fn = p_stats_b[pid][s]['tp'], p_stats_b[pid][s]['fp'], p_stats_b[pid][s]['fn']
            res_b[s]['iou'].append(tp / max(1, tp+fp+fn) * 100)
            res_b[s]['dice'].append(2*tp / max(1, 2*tp+fp+fn) * 100)
            
    final_res = {}
    for s in seeds_to_ensemble:
        final_res[s] = {
            'A': {'iou': np.mean(res_a[s]['iou']), 'dice': np.mean(res_a[s]['dice'])},
            'B': {'iou': np.mean(res_b[s]['iou']), 'dice': np.mean(res_b[s]['dice'])}
        }
    return final_res

print("\n=== OPTIMIZED EVALUATION RESULTS (SIZE 25, THRESH 0.3) ===")
for arm in ["Frozen Baseline", "Untrained Head"]:
    print(f"\nARM: {arm}")
    seeds = [0] if arm == "Frozen Baseline" else [9]
    res = evaluate_patient(arm, seeds)
    
    for s in seeds:
        print(f"  Seed {s} - Soft Pipeline (A): {res[s]['A']['iou']:.2f}% IoU / {res[s]['A']['dice']:.2f}% Dice")
        print(f"  Seed {s} - Hard Pipeline (B): {res[s]['B']['iou']:.2f}% IoU / {res[s]['B']['dice']:.2f}% Dice")
print("========================================================\n")
