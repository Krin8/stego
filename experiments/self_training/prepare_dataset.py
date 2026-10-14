import numpy as np
import pydicom
import json
from pathlib import Path
from scipy import ndimage as ndi
import torch
from skimage.transform import resize
from runtime import load_image, inverse_grid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / 'audit/naming'

def prepare():
    rr = json.loads((BASE / 'image_records.json').read_text())
    ids = [i for i, r in enumerate(rr) if r['split'] == 'train']
    
    features = np.load(ROOT / 'audit/improve/B2_448_features.npy', mmap_mode='r')
    centers = np.load(BASE / 'inductive_s0_centers.npy')
    
    N = len(ids)
    images = np.zeros((N, 1, 256, 256), dtype=np.float32)
    labels = np.zeros((N, 256, 256), dtype=np.int64)
    
    print(f"Preparing dataset with {N} samples...")
    for j, idx in enumerate(ids):
        f = features[idx].reshape(784, 768)
        d = np.linalg.norm(f[:, None] - centers, axis=-1)
        assignment = d.argmin(axis=-1).reshape(28, 28)
        native = inverse_grid(assignment, rr[idx], outside=8)
        
        hu = load_image(rr[idx])
        
        # Resize to 256x256
        images[j, 0] = resize(hu, (256, 256), order=1, preserve_range=True, anti_aliasing=False)
        labels[j] = resize(native, (256, 256), order=0, preserve_range=True, anti_aliasing=False)
        
        if j % 200 == 0:
            print(f"Processed {j}/{N}")
            
    np.save(HERE / 'train_images.npy', images)
    np.save(HERE / 'train_targets.npy', labels)
    print("Dataset prepared!")

if __name__ == '__main__':
    prepare()
