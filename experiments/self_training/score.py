import numpy as np
import json
from pathlib import Path

def score():
    import sys
    sys.path.append(str(Path(__file__).resolve().parent))
    import runtime
    runtime.STAGE = 'evaluation'
    HERE = Path(__file__).resolve().parent
    ROOT = HERE.parents[1]
    BASE = ROOT / 'audit/naming'
    
    rr = json.loads((BASE / 'image_records.json').read_text())
    val_ids = [i for i, r in enumerate(rr) if r['split'] == 'val']
    
    gt = np.load(ROOT / 'audit/improve/scoring_masks_native.npy', mmap_mode='r')
    pred = np.lib.format.open_memmap(HERE / 'predictions/raw_clusters_native.npy', mode='r')
    
    # The original ground truth class for liver is 1
    
    # From naming.json, N3 selects cluster 2 as Liver.
    liver_clusters = [2]
        
    print(f"Liver clusters are: {liver_clusters}")
    
    tp = fp = fn = 0
    for j, idx in enumerate(val_ids):
        gt_liver = (gt[idx] == 1)
        p = pred[j]
        pred_liver = np.isin(p, liver_clusters)
        
        intersection = np.logical_and(gt_liver, pred_liver).sum()
        tp += intersection
        fp += pred_liver.sum() - intersection
        fn += gt_liver.sum() - intersection
        
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    print(f"Self-Training FCN IoU: {iou * 100:.2f}%")

if __name__ == '__main__':
    score()
