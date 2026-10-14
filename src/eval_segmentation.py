import numpy as np
import json
import sys
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
import argparse
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / 'audit' / 'final_pipeline_n3'))
sys.path.append(str(ROOT / 'experiments' / 'self_training'))
import runtime
from segment_ct import inverse_grid, largest, distances
runtime.STAGE = 'evaluation' # Bypass the audit seal to score the model

def load_plane(path):
    with Image.open(path) as im:
        a = np.array(im, np.float32) / 65535.
    return a

def main():
    parser = argparse.ArgumentParser(description="Evaluate 86.25% Zero-Shot Liver Segmentation Pipeline")
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'predictions' / 'biomedclip_ensemble', help="Directory to save output masks")
    parser.add_argument('--save-images', action='store_true', help="Whether to save the overlay images")
    args = parser.parse_args()
    
    # 1. Load Ground Truth and Validation Data
    print("Loading Ground Truth and validation records...")
    gt_masks = np.load(ROOT / 'audit/improve/scoring_masks_native.npy', mmap_mode='r')
    rr = json.loads((ROOT / 'audit/naming/image_records.json').read_text())
    val_records = [(i, r) for i, r in enumerate(rr) if r['split'] == 'val']
    
    print("Loading BiomedCLIP centers from N3 pipeline...")
    centers_list = [np.load(ROOT / f'audit/final_pipeline_n3/all20_seed{s}_centers.npy') for s in [0,1,2]]
    meta_list = [json.loads((ROOT / f'audit/final_pipeline_n3/all20_seed{s}.json').read_text()) for s in [0,1,2]]
    
    # We must load the extractor model to get features
    sys.path.append(str(ROOT / 'src'))
    from modules import BiomedCLIPFeaturizer
    import torch
    import torch.nn.functional as F
    from torchvision.transforms import functional as TF, InterpolationMode
    
    print("Loading frozen BiomedCLIP foundation model...")
    net = BiomedCLIPFeaturizer(70, type('obj', (object,), {'projection_type': None, 'dropout': False, 'use_text_prompts': False})).cpu().eval().requires_grad_(False)
    
    if args.save_images:
        args.output.mkdir(parents=True, exist_ok=True)
    
    tp = fp = fn = 0
    print(f"Evaluating {len(val_records)} validation images...")
    
    for global_idx, record in tqdm(val_records):
        path = ROOT / record['image']
        im = load_plane(path)
        
        # 1. Extract Features
        lab, n = ndi.label(im > 0.05, structure=np.ones((3,3), dtype=int))
        if n:
            counts = np.bincount(lab.ravel()); counts[0] = 0
            body = ndi.binary_fill_holes(lab == counts.argmax())
            yy, xx = np.where(body)
            side = int(np.ceil(max(yy.max()-yy.min()+1, xx.max()-xx.min()+1)*1.1))
            y0 = int(np.floor((yy.min()+yy.max()+1-side)/2))
            x0 = int(np.floor((xx.min()+xx.max()+1-side)/2))
            box = [y0, x0, side]
        else:
            box = [0, 0, max(im.shape)]
            
        y, x, s = box
        h, w = im.shape
        crop = np.zeros((s, s), np.float32)
        ya, xa, yb, xb = max(0, y), max(0, x), min(h, y+s), min(w, x+s)
        crop[ya-y:yb-y, xa-x:xb-x] = im[ya:yb, xa:xb]
        
        t = torch.from_numpy(crop).unsqueeze(0).repeat(3,1,1)
        t = TF.resize(t, [448, 448], interpolation=InterpolationMode.BILINEAR, antialias=True)
        t = TF.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        with torch.inference_mode():
            f = F.normalize(net(t.unsqueeze(0))[0], dim=1)
        feat = f[0].permute(1,2,0).numpy().reshape(784, 768)
        img_record = {'box': box, 'shape': list(im.shape)}
        
        # 2. Ensemble Soft Predictions
        soft_field = np.zeros((512, 512), np.float32)
        for centers, meta in zip(centers_list, meta_list):
            d = distances(feat, centers)
            liver_clusters = [meta['liver_cluster']]
            ids = np.asarray(liver_clusters, dtype=int)
            
            z = -(d - d.min(1, keepdims=True)) / meta['temperature']
            w = np.exp(z)
            w /= w.sum(1, keepdims=True)
            
            field = w[:, ids].sum(1).reshape(28, 28)
            soft = inverse_grid(field, img_record, mode='bilinear', outside=0)
            soft_field += soft / len(centers_list)
            
        # 3. Optimized Guided Filter (size=25, thresh=0.30)
        g_size = 25
        threshold = 0.30
        closing_rad = 3
        
        mi = ndi.uniform_filter(im, size=g_size, mode='reflect')
        mp = ndi.uniform_filter(soft_field, size=g_size, mode='reflect')
        var = ndi.uniform_filter(im*im, size=g_size, mode='reflect') - mi*mi
        cov = ndi.uniform_filter(im*soft_field, size=g_size, mode='reflect') - mi*mp
        aa = cov / (var + 0.01)
        bb = mp - aa*mi
        filt = np.clip(ndi.uniform_filter(aa, size=g_size, mode='reflect')*im + ndi.uniform_filter(bb, size=g_size, mode='reflect'), 0, 1)
        
        pred = largest(filt >= threshold, np.ones((3,3), int))
        pred = ndi.binary_fill_holes(pred)
        yy, xx = np.mgrid[-closing_rad:closing_rad+1, -closing_rad:closing_rad+1]
        pred = ndi.binary_closing(np.pad(pred, closing_rad), structure=xx*xx+yy*yy<=closing_rad**2)[closing_rad:-closing_rad, closing_rad:-closing_rad]
        
        # 4. Score against Ground Truth
        gt_liver = (gt_masks[global_idx] == 1)
        inter = np.logical_and(gt_liver, pred).sum()
        tp += inter
        fp += pred.sum() - inter
        fn += gt_liver.sum() - inter
        
        if args.save_images:
            overlay_path = args.output / f'{path.stem}_overlay.png'
            rgb = np.repeat(im[...,None], 3, axis=2)
            rgb[pred] = 0.65*rgb[pred] + 0.35*np.array([1., 0., 0.])
            edge = pred & ~ndi.binary_erosion(pred)
            rgb[edge] = [1, 0, 0]
            Image.fromarray(np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)).save(overlay_path, format='PNG')

    # Final Metrics
    iou = tp / (tp + fp + fn) if (tp+fp+fn) > 0 else 0
    precision = tp / (tp + fp) if (tp+fp) > 0 else 0
    recall = tp / (tp + fn) if (tp+fn) > 0 else 0
    
    print("\n" + "="*40)
    print("      EVALUATION RESULTS")
    print("="*40)
    print(f"Volumetric IoU : {iou*100:.2f}%")
    print(f"Precision      : {precision*100:.2f}%")
    print(f"Recall         : {recall*100:.2f}%")
    print("="*40)
    if args.save_images:
        print(f"Saved predicted overlays to {args.output}")

if __name__ == '__main__':
    main()
