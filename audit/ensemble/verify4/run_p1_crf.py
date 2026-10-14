import sys
import os
import json
import numpy as np
import pydicom
from pathlib import Path
from scipy import ndimage as ndi
from sklearn.cluster import KMeans
import torch
import torch.nn as nn

ROOT = Path('/Users/navneetbavineni/STEGO').resolve()
sys.path.insert(0, str(ROOT / 'audit' / 'final_pipeline_n3'))
sys.path.insert(0, str(ROOT / 'experiments' / 'stego_colocated'))

import runtime as rt
rt.PACKAGE = ROOT
rt.STAGE = 'evaluation'

sys.path.insert(0, str(ROOT / 'src'))
from crf import dense_crf
import torchvision.transforms.functional as VF

from runtime import records, seed_all, load_image, frozen_backbone, image_tensor, crop_image, distances, inverse_grid, resize224
from segment_ct import guided, largest

rr = records()
train_ids = [i for i, r in enumerate(rr) if r['split'] == 'train']
val_ids = [i for i, r in enumerate(rr) if r['split'] == 'val']

features = np.load(ROOT / "audit/improve/B2_448_features.npy", mmap_mode='r')
val_y = {512: np.load(rt.OLD / 'scoring_masks_native.npy', mmap_mode='r')}

def embedding(i):
    f = torch.from_numpy(np.array(features[i])).reshape(28,28,768).permute(2,0,1)[None]
    with torch.no_grad():
        z = f[0].permute(1,2,0).contiguous().numpy()
    return z

class StegoHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.cluster2 = nn.Sequential(
            nn.Conv2d(768, 768, 1),
            nn.ReLU(),
            nn.Conv2d(768, 70, 1)
        )
    def embed(self, x):
        return self.cluster2(x)

def apply_postprocessing(field, im, r):
    """
    field: probability of liver (not used directly for CRF, we use logits instead)
    im: numpy array (H, W)
    """
    # This is handled directly in the loop now.
    pass

s = 0
print(f"Running Trained P1 Head Seed {s} using ONLY N2 (Area) Naming and Optimized Post-Processing...")
seed_all(s)
model = StegoHead().eval().requires_grad_(False)
model.load_state_dict(torch.load(ROOT / f"experiments/stego_colocated/P1/stage1/checkpoints/seed{s}/final.pt", map_location='cpu', weights_only=True)['state_dict'], strict=False)
train_features = np.stack([embedding(i) for i in train_ids])
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
G = 28
yy,xx = np.mgrid[-5:6,-5:6]; disk = xx*xx+yy*yy<=25

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
    means[empty]=hu[iy[empty],ix[empty]]
    
    z = z_train[n*784:(n+1)*784]; d_dist = distances(z, centers); assignment = d_dist.argmin(1)
    nx = (cx-xs.min())/max(1,xs.max()-xs.min()); ny = (cy-ys.min())/max(1,ys.max()-ys.min())
    for cluster in range(8):
        u = use[assignment[use]==cluster]
        sums[cluster,0] += len(u)
        sums[cluster,1:] += [nx[u].sum(), ny[u].sum(), means[u].sum()]

stats = []
for k in range(8):
    count = sums[k,0]
    x,y,h = sums[k,1:]/count if count else [float(1),float(1),float(0)]
    stats.append(dict(cluster=k, area=float(count/max(1,sums[:,0].sum())), candidate=bool(count and x<.5 and y<.6 and 40<=h<=180)))

possible = [x for x in stats if x['candidate']]
n2 = max(possible, key=lambda x:x['area'])['cluster'] if possible else None

print(f"N2 selected cluster: {n2}")

val_feats = np.stack([embedding(i).reshape(28*28, 768) for i in val_ids])
f_val = torch.from_numpy(val_feats).reshape(len(val_feats), 28, 28, 768).permute(0,3,1,2)
out_val = []
for chunk in torch.split(f_val, 32):
    with torch.no_grad():
        out_val.append(model.embed(chunk).permute(0,2,3,1).reshape(len(chunk), 784, 70).numpy())
val_embeds = np.concatenate(out_val, axis=0)

tp_total = 0; fp_total = 0; fn_total = 0

from tqdm import tqdm
for i_idx, i in enumerate(tqdm(val_ids, desc="Evaluating all 472 slices")):
    z = val_embeds[i_idx]
    im = load_image(rr[i])
    d_val = distances(z, centers)
    
    # Calculate Liver Logits
    liver_logits = -(d_val[:, n2]) / temp if n2 is not None else np.zeros(784)
    # Background Logits (approximation: max of all other clusters to simulate logits before softmax)
    bg_clusters = [k for k in range(8) if k != n2]
    bg_logits = np.max(-(d_val[:, bg_clusters]) / temp, axis=1) if len(bg_clusters) > 0 else np.zeros(784)
    
    # Reshape and Upsample to 512x512
    liver_grid = liver_logits.reshape(G,G)
    bg_grid = bg_logits.reshape(G,G)
    
    liver_soft = inverse_grid(liver_grid, rr[i], mode='bilinear', outside=-100)
    bg_soft = inverse_grid(bg_grid, rr[i], mode='bilinear', outside=0)
    
    # Custom Guided Filter with size=25 (from the flowchart)
    liver_prob = np.exp(liver_soft) / (np.exp(liver_soft) + np.exp(bg_soft) + 1e-6)
    
    # Custom inline guided filter implementation to support size override
    def guided_custom(image, field, size=25):
        import scipy.ndimage as ndi
        mi = ndi.uniform_filter(image, size=size, mode='reflect')
        mp = ndi.uniform_filter(field, size=size, mode='reflect')
        var = ndi.uniform_filter(image*image, size=size, mode='reflect') - mi*mi
        cov = ndi.uniform_filter(image*field, size=size, mode='reflect') - mi*mp
        aa = cov / (var + 0.01)
        bb = mp - aa*mi
        return np.clip(ndi.uniform_filter(aa, size=size, mode='reflect')*image + ndi.uniform_filter(bb, size=size, mode='reflect'), 0, 1)
    
    guided_prob = guided_custom(im, liver_prob, size=25)
    
    # Threshold at 0.30 (from flowchart)
    pred_a = largest(guided_prob >= 0.30, np.ones((3,3), int))
    pred_a = ndi.binary_fill_holes(pred_a)
    
    # 3-radius disk closing
    yy,xx = np.mgrid[-3:4,-3:4]
    pred_a = ndi.binary_closing(np.pad(pred_a, 3), structure=xx*xx+yy*yy<=9)[3:-3, 3:-3]
    gt = (val_y[512][i] == 1)
    
    tp_total += np.logical_and(pred_a, gt).sum()
    fp_total += pred_a.sum() - np.logical_and(pred_a, gt).sum()
    fn_total += gt.sum() - np.logical_and(pred_a, gt).sum()

iou = tp_total / max(1, tp_total + fp_total + fn_total) * 100

total_pixels = len(val_ids) * 512 * 512
bg_tp = total_pixels - tp_total - fp_total - fn_total
bg_fp = fn_total
bg_fn = fp_total
bg_iou = bg_tp / max(1, bg_tp + bg_fp + bg_fn) * 100

miou = (iou + bg_iou) / 2

print(f"\n=== N2 ONLY RESULT ===")
print(f"Liver IoU (Foreground): {iou:.2f}%")
print(f"Background IoU: {bg_iou:.2f}%")
print(f"mIoU (Mean): {miou:.2f}%")
print("======================\n")
