import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf

# Add src to path so we can import models and data
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from modules import BiomedCLIPFeaturizer
from data import NegativeImageDataset
from chaos_preprocessing import get_chaos_transform
from torch.utils.data import DataLoader
import logging

def main():
    logging.getLogger("pydicom").setLevel(logging.CRITICAL)
    logging.getLogger("pydicom.pixels.decoders.base").setLevel(logging.CRITICAL)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Initialize BiomedCLIP via STEGO module
    cfg = OmegaConf.create({
        "arch": "biomedclip",
        "continuous": False,
        "dim": 70,
        "extra_classes": 0,
        "n_class": 8,
        "pretrained_weights": None,
        "projection_type": "nonlinear",
        "use_salience": False,
        "dropout": False
    })
    model = BiomedCLIPFeaturizer(cfg.dim, cfg).to(device).eval().requires_grad_(False)
    
    # 2. Load LungCT Negatives Dataset
    res = 448
    grid = res // 16
    dim = 768
    
    # The transform must match B2_448 (448 res, no augmentations)
    from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
    from PIL import Image
    import glob
    import pydicom

    class LungDataset(torch.utils.data.Dataset):
        def __init__(self, root):
            self.files = sorted(glob.glob(f"{root}/**/*.dcm", recursive=True))
            self.transform = Compose([
                Resize(res, Image.BILINEAR),
                CenterCrop(res),
                ToTensor(),
                Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ])
        def __len__(self):
            return len(self.files)
        def __getitem__(self, idx):
            try:
                ds = pydicom.dcmread(self.files[idx], force=True)
                arr = ds.pixel_array
                if arr.ndim != 2:
                    raise ValueError("Not a 2D image")
            except:
                arr = np.zeros((512, 512), dtype=np.float32)
            
            # CT Window normalization (-150 to 250)
            arr = np.asarray(arr, dtype=np.float32)
            arr = np.clip((arr - -150.0) / (250.0 - -150.0), 0, 1)
            # Convert to 3 channel RGB (BiomedCLIP expects RGB)
            arr = (arr * 255).astype(np.uint8)
            img = Image.fromarray(arr).convert("RGB")
            return self.transform(img)

    dataset = LungDataset("Dataset/LungCT_Negatives")
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    
    num_lungs = len(dataset)
    print(f"Found {num_lungs} Lung CT slices.")
    
    # 3. Create Memory-Mapped Array
    out_dir = Path(__file__).parent.parent / "improve"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "Lung_448_features.npy"
    
    dest = np.lib.format.open_memmap(out_path, mode='w+', dtype='float32', shape=(num_lungs, grid*grid, dim))
    
    # 4. Extract and Normalize
    print("Extracting features...")
    start = 0
    with torch.inference_mode():
        for batch in tqdm(loader):
            # batch is (N, C, H, W)
            x = batch.to(device)
            # Forward pass through STEGO backbone (returns feats, code)
            f, _ = model(x) # f shape: (N, 768, 28, 28)
            
            # Match B2_448 normalization and shape
            f = F.normalize(f, dim=1) # L2 normalize along channel dimension
            
            # Permute to (N, 28*28, 768)
            f_flat = f.permute(0, 2, 3, 1).reshape(-1, grid*grid, dim)
            
            dest[start:start+x.shape[0]] = f_flat.cpu().numpy()
            start += x.shape[0]
            
    dest.flush()
    print(f"Extraction complete! Saved to {out_path} (Shape: {dest.shape})")

if __name__ == '__main__':
    main()
