import sys
import os
import numpy as np
import pydicom
from pathlib import Path
from scipy import ndimage as ndi
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path('/Users/navneetbavineni/STEGO').resolve()
sys.path.insert(0, str(ROOT / 'audit' / 'final_pipeline_n3'))
sys.path.insert(0, str(ROOT / 'experiments' / 'stego_colocated'))

import runtime as rt
rt.PACKAGE = ROOT
rt.STAGE = 'evaluation'

from runtime import records, seed_all, load_image, frozen_backbone, crop_image, distances, inverse_grid, resize224
from segment_ct import guided, largest

ARTIFACT_DIR = ROOT / 'audit' / 'ensemble' / 'verify4' / 'n2_visualizations'
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
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
        self.head = nn.Sequential(
            nn.Conv2d(768, 768, 1),
            nn.ReLU(),
            nn.Conv2d(768, 70, 1)
        )
    def embed(self, x):
        return self.head(x)

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
    return soft, filt, pred

s = 9
seed_all(s)
model = StegoHead().eval().requires_grad_(False)
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

val_feats = np.stack([embedding(i).reshape(28*28, 768) for i in val_ids])
f_val = torch.from_numpy(val_feats).reshape(len(val_feats), 28, 28, 768).permute(0,3,1,2)
out_val = []
for chunk in torch.split(f_val, 32):
    with torch.no_grad():
        out_val.append(model.embed(chunk).permute(0,2,3,1).reshape(len(chunk), 784, 70).numpy())
val_embeds = np.concatenate(out_val, axis=0)

for i_idx, i in enumerate(val_ids):
    z = val_embeds[i_idx]
    im = load_image(rr[i])
    d_val = distances(z, centers)
    
    weights = np.exp(-(d_val-d_val.min(1,keepdims=True))/temp)
    weights /= weights.sum(1,keepdims=True)
    field = weights[:,n2].reshape(G,G) if n2 is not None else np.zeros((G,G),np.float32)
    
    soft, filt, pred_a = apply_postprocessing(field, im, rr[i])
    gt = (val_y[512][i] == 1)
    
    tp = np.logical_and(pred_a, gt).sum()
    fp = pred_a.sum() - tp
    fn = gt.sum() - tp
    iou = tp / max(1, tp + fp + fn) * 100
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    patient_id = rr[i]['patient']
    slice_id = str(rr[i]['dicom']).split('/')[-1].replace('.dcm', '')
    title = f"{patient_id} - Slice {slice_id} (IoU: {iou:.1f}%)"
    fig.suptitle(title, fontsize=16)
    
    axes[0].imshow(im, cmap='gray')
    axes[0].set_title("Original Input")
    axes[0].axis('off')
    
    axes[1].imshow(im, cmap='gray')
    axes[1].imshow(gt, cmap='jet', alpha=0.5)
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis('off')
    
    axes[2].imshow(im, cmap='gray')
    axes[2].imshow(pred_a, cmap='jet', alpha=0.5)
    axes[2].set_title("Final Clustered Output")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / f"n2_eval_{i_idx:02d}_{patient_id}_{slice_id}.png")
    plt.close()

print("Visualizations generated!")
