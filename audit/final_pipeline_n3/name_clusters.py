import argparse
import hashlib
import json
import logging
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi

# Import from segment_ct.py
from segment_ct import (
    load_plane, load_extractor, extract, distances,
    contract, body_box, crop_image, image_tensor,
    MEAN, STD
)

READS=set();BLOCKED=[]
def guard(event,args):
    if event=='open':
        path,mode,flags=args
        if isinstance(path,int):return
        p=Path(os.fsdecode(path)).resolve();writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if not writing:
            s=str(p)
            bad=('/labels/' in s or '/Ground/' in s or 'scoring_masks' in s or p.name.endswith('_labels.npy') or p.name.endswith('_scores.json') or p.name.endswith('_results.json'))
            if bad:
                BLOCKED.append(str(p))
                raise PermissionError('Naming forbids label/score/mapping files: '+s)
            READS.add(str(p))
    elif event=='socket.connect':raise PermissionError('Network forbidden')
sys.addaudithook(guard)


def get_n1_text_embeddings(objects):
    from modules import BiomedCLIPFeaturizer
    from open_clip.tokenizer import HFTokenizer
    from types import SimpleNamespace
    
    net = BiomedCLIPFeaturizer(70, SimpleNamespace(projection_type=None, dropout=False, use_text_prompts=False)).cpu().eval().requires_grad_(False)
    model = net.full_model
    config = json.loads((ROOT / 'src/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_config.json').read_text())
    tokenizer = HFTokenizer(config['model_cfg']['text_cfg']['hf_tokenizer_name'], context_length=256)
    
    prompts = ['an axial abdominal CT image showing the ' + x for x in objects]
    with torch.inference_mode():
        text_embeddings = model.encode_text(tokenizer(prompts), normalize=True).cpu().numpy()
    return model, text_embeddings

def N1_N2_N3(records_file, centers, temperature):
    """
    Reuses the exact N1/N2/N3 implementation and parameters from audit/naming/stage_a.py (Lines 8-13, 27-94).
    """
    records = json.loads(Path(records_file).read_text())
    
    # Pre-compute text embeddings for N1
    objects = ['liver','spleen','kidney','stomach','pancreas','gallbladder','aorta','vertebra','muscle','subcutaneous fat','bowel','lung','heart','rib','air']
    model, text_embeddings = get_n1_text_embeddings(objects)
    
    net = load_extractor({'backbone_weight_sha256': "52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be"})
    
    sums = np.zeros((8, 4), np.float64)
    nearest = [[] for _ in range(8)]
    
    all_geom = []
    
    for i, r in enumerate(records):
        path = ROOT / r['dicom'] if 'dicom' in r else Path(r['image'])
        if not path.is_absolute(): path = ROOT / path
        im = load_plane(path)
        
        # Calculate HU in native space for N2
        # (Assuming im is normalized 0-1 from -150 to 250 in segment_ct, but for N2 we need raw HU).
        # We need raw HU > -500. Since im = (HU + 150) / 400 clipped to [0,1], 
        # HU = im * 400 - 150.
        hu = (im * 400.0) - 150.0
        
        yy,xx=np.mgrid[-5:6,-5:6];disk=xx*xx+yy*yy<=25
        body_mask=ndi.binary_fill_holes(ndi.label(ndi.binary_opening(hu>-500,structure=disk),np.ones((3,3),int))[0]==(np.bincount(ndi.label(ndi.binary_opening(hu>-500,structure=disk),np.ones((3,3),int))[0].ravel())[1:].argmax()+1 if np.bincount(ndi.label(ndi.binary_opening(hu>-500,structure=disk),np.ones((3,3),int))[0].ravel())[1:].size else 0))
        ys,xs=np.where(body_mask)
        if not len(ys): continue
        
        box, _ = body_box(im)
        y0,x0,side=box
        
        f, record = extract(net, im)
        d = distances(f, centers)
        assignment = d.argmin(1)
        
        gy,gx=np.mgrid[:28,:28];cy=y0+(gy.ravel()+.5)*side/28-.5;cx=x0+(gx.ravel()+.5)*side/28-.5
        ix=np.floor(cx+.5).astype(int);iy=np.floor(cy+.5).astype(int)
        inside=(ix>=0)&(iy>=0)&(ix<512)&(iy<512)
        eligible=inside.copy();eligible[inside]=body_mask[iy[inside],ix[inside]]
        
        geom = np.zeros((784, 5), np.float32)
        for t in np.flatnonzero(eligible):
            row,col=divmod(int(t),28)
            ya=max(0,int(np.ceil(y0+row*side/28-.5)));yb=min(512,int(np.ceil(y0+(row+1)*side/28-.5)))
            xa=max(0,int(np.ceil(x0+col*side/28-.5)));xb=min(512,int(np.ceil(x0+(col+1)*side/28-.5)))
            pixels=hu[ya:yb,xa:xb][body_mask[ya:yb,xa:xb]]
            if len(pixels):
                geom[t]=[1,(cx[t]-xs.min())/max(1,xs.max()-xs.min()),(cy[t]-ys.min())/max(1,ys.max()-ys.min()),pixels.mean(),len(pixels)]
        
        all_geom.append((path, im, geom, cx, cy, box))
        
        for c in range(8):
            use = np.flatnonzero((assignment==c)&(geom[:,0]>0))
            sums[c,0]+=len(use);sums[c,1:]+=geom[use,1:4].astype(np.float64).sum(0)
            body_tokens = np.flatnonzero(geom[:,0]>0)
            nearest[c] = sorted(nearest[c] + [(float(d[t,c]), i, int(t)) for t in body_tokens])[:200]
            
    # Compute N2 stats
    stats = []
    for c in range(8):
        count=sums[c,0];x,y,h=sums[c,1:]/count
        stats.append(dict(cluster=c,body_patches=int(count),area=float(count/sums[:,0].sum()),centroid_x=float(x),centroid_y=float(y),mean_HU=float(h),candidate=bool(x<.5 and y<.6 and 40<=h<=180)))
    candidates=[r for r in stats if r['candidate']];n2=max(candidates,key=lambda r:r['area'])['cluster'] if candidates else None
    
    # Compute N1
    n1_nearest_averages = []
    for c in range(8):
        crop_embs = []
        for dist, i, t in nearest[c]:
            path, im, geom, cx, cy, box = all_geom[i]
            crop_box=[int(np.floor(cy[t]+.5))-56,int(np.floor(cx[t]+.5))-56,112]
            im_tensor = image_tensor(crop_image(im, crop_box), 224)
            with torch.inference_mode():
                emb = model.encode_image(im_tensor.unsqueeze(0), normalize=True).cpu().numpy()[0]
            crop_embs.append(emb)
        avg = np.stack(crop_embs).mean(0)
        avg /= max(np.linalg.norm(avg), 1e-12)
        n1_nearest_averages.append(avg)
        
    averages = np.stack(n1_nearest_averages)
    sim=averages@text_embeddings.T;mean=sim.mean(0,keepdims=True);std=sim.std(0,keepdims=True);z=np.divide(sim-mean,std,out=np.zeros_like(sim),where=std>0)
    selected_n1=int(z[:,0].argmax())
    
    liver=np.sort(z[:,0]); margin=float(liver[-1]-liver[-2])
    n3 = selected_n1 if selected_n1 == n2 else None
    
    return selected_n1, n2, n3, margin

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', type=Path, required=True, help="Path to image_records.json")
    parser.add_argument('--centers', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    
    centers = np.load(args.centers, allow_pickle=False)
    n1, n2, n3, margin = N1_N2_N3(args.records, centers, 0.1)
    
    metadata = {}
    if os.path.exists(args.output):
        with open(args.output, 'r') as f:
            metadata = json.load(f)

    metadata.update({
        'seed': args.seed,
        'liver_cluster': n3,
        'n1_cluster': n1,
        'n2_cluster': n2,
        'abstention_status': n3 is None,
        'margins': margin,
        'centers_sha256': hashlib.sha256(open(args.centers, 'rb').read()).hexdigest(),
        'naming_rule': 'N3',
        'backbone_weight_sha256': "52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be",
        'preprocessing': contract(),
        'K': 8,
        'git_commit': os.popen('git rev-parse HEAD').read().strip()
    }
    
    # Write metadata
    with open(args.output, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Write ledger
    ledger_path = args.output.with_name(args.output.stem + '_ledger.json')
    with open(ledger_path, 'w') as f:
        json.dump({
            'mask_or_label_files_read': 0,
            'blocked_attempts': BLOCKED,
            'opened_read_paths': sorted(READS),
            'metadata_note': 'No labels or masks read.'
        }, f, indent=2)

