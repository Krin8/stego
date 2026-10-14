import numpy as np
import pydicom
from pathlib import Path
from scipy import ndimage as ndi
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / 'audit/naming'

def records():
    rr = json.loads((BASE / 'image_records.json').read_text())
    return rr

def largest(mask, structure):
    labels, _ = ndi.label(mask, structure)
    if not labels.any(): return mask
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == counts.argmax()

def create_masks():
    rr = records()
    ids = [i for i, r in enumerate(rr) if r['split'] == 'train']
    G = 28
    
    masks = np.zeros((len(ids), G, G), dtype=np.float32)
    
    yy, xx = np.mgrid[-5:6, -5:6]
    disk = xx*xx + yy*yy <= 25
    
    for j, i in enumerate(ids):
        r = rr[i]
        ds = pydicom.dcmread(r['dicom'])
        hu = ds.pixel_array.astype(np.float32) * float(getattr(ds, 'RescaleSlope', 1)) + float(getattr(ds, 'RescaleIntercept', 0))
        body = ndi.binary_fill_holes(largest(ndi.binary_opening(hu > -500, structure=disk), np.ones((3,3), int)))
        
        y0, x0, side = r['box']
        gy, gx = np.mgrid[:G, :G]
        cy = y0 + (gy.ravel() + .5) * side / G - .5
        cx = x0 + (gx.ravel() + .5) * side / G - .5
        iy = np.floor(cy + .5).astype(int)
        ix = np.floor(cx + .5).astype(int)
        
        eligible = (iy >= 0) & (ix >= 0) & (iy < 512) & (ix < 512)
        valid = np.flatnonzero(eligible)
        eligible[valid] = body[iy[valid], ix[valid]]
        
        masks[j] = eligible.reshape(G, G).astype(np.float32)
        
        if j % 200 == 0:
            print(f"Processed {j}/{len(ids)}")
            
    np.save('body_masks.npy', masks)
    print("Saved body_masks.npy")

if __name__ == '__main__':
    create_masks()
