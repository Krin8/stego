import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# Setup for testing
sys.path.append(str(HERE))
import runtime as rt
import model
import reference_loss

def get_step0_score():
    print("Testing Step 0 Identity Head...")
    # This would require running a full predict/evaluate pass.
    # For now, let's just note we need to run it.
    pass

def test_preflight_fractions():
    print("Computing attraction fractions for body-masked sampling...")
    # Load 200 slice PREFLIGHT sample
    records = rt.records()
    train_recs = [r for r in records if r['split'] == 'train']
    # Just take first few patients for 200 slices
    sample_ids = [i for i, r in enumerate(records) if r['split'] == 'train'][:200]
    
    body_masks = np.load(HERE / 'body_masks.npy', mmap_mode='r')
    
    # Let's load features
    corr_loss = reference_loss.CorrespondenceLoss()
    
    # We need to simulate the loss's correlation matrix
    total_intra = 0; total_knn = 0; total_rand = 0
    pos_intra = 0; pos_knn = 0; pos_rand = 0
    
    # shifts
    b_self = 0.18
    b_knn = 0.12
    b_rand = 0.46
    
    for idx in sample_ids:
        # Load features (28x28)
        feat = rt.load_features(records[idx])
        feat = torch.from_numpy(feat).cuda() # (768, 28, 28)
        
        # Load mask (224x224) and resize to 28x28
        mask = torch.from_numpy(body_masks[idx]).float().cuda().unsqueeze(0).unsqueeze(0)
        mask_28 = F.interpolate(mask, size=(28, 28), mode='nearest').squeeze().bool()
        
        # Flatten
        C = feat.shape[0]
        feat_flat = feat.view(C, -1).t() # (784, 768)
        mask_flat = mask_28.view(-1)     # (784,)
        
        # Filter to body only
        feat_body = feat_flat[mask_flat] # (N_body, 768)
        N = feat_body.shape[0]
        if N < 2: continue
        
        # Normalize
        feat_body = F.normalize(feat_body, p=2, dim=-1)
        
        # Compute correlations
        corr = torch.matmul(feat_body, feat_body.t()) # (N, N)
        
        # Intra (self): since we don't have separate augmented views easily, we simulate self correlation
        # by looking at spatial neighbors or just the raw correlations.
        # Actually, PREFLIGHT just takes features and pairs them.
        # Let's just look at the raw correlation distribution.
        
        # kNN and Rand:
        # In STEGO, it computes correlations between feature maps.
        pass

if __name__ == '__main__':
    test_preflight_fractions()
