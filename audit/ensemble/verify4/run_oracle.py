import sys
import os
import json
import numpy as np
import pydicom
from pathlib import Path
from scipy import ndimage as ndi
from sklearn.cluster import KMeans

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

arm = "Untrained Head"
s = 9
print(f"Running Oracle Evaluation on {arm} Seed {s} (size=25, thresh=0.30)...")

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
G = 28
val_embeds = get_val_embeddings(arm, s)

best_iou = 0
best_cluster = -1

for test_cluster in range(8):
    tp_total = 0
    fp_total = 0
    fn_total = 0
    
    for i_idx, i in enumerate(val_ids):
        z = val_embeds[i_idx]
        im = load_image(rr[i])
        d_val = distances(z, centers)
        
        weights = np.exp(-(d_val-d_val.min(1,keepdims=True))/temp)
        weights /= weights.sum(1,keepdims=True)
        field = weights[:,test_cluster].reshape(G,G)
        
        pred_a = apply_postprocessing(field, im, rr[i])
        gt = (val_y[512][i] == 1)
        
        tp_total += np.logical_and(pred_a, gt).sum()
        fp_total += pred_a.sum() - np.logical_and(pred_a, gt).sum()
        fn_total += gt.sum() - np.logical_and(pred_a, gt).sum()
        
    iou = tp_total / max(1, tp_total + fp_total + fn_total) * 100
    print(f"Cluster {test_cluster}: {iou:.2f}% IoU")
    if iou > best_iou:
        best_iou = iou
        best_cluster = test_cluster

print(f"\n=== ORACLE RESULT ===")
print(f"Best Cluster: {best_cluster}")
print(f"Oracle Max IoU: {best_iou:.2f}%")
print("=====================\n")
