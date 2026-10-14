import sys
import torch
from pathlib import Path
from omegaconf import OmegaConf
from modules import BiomedCLIPFeaturizer

def main():
    print("Testing BiomedCLIP Connection...")
    
    cfg = OmegaConf.load(Path(__file__).parent / "configs" / "train_config.yaml")
    
    # 2. Instantiate Featurizer
    try:
        model = BiomedCLIPFeaturizer(dim=cfg.dim, cfg=cfg)
        print("✅ BiomedCLIPFeaturizer initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return

    # 3. Test Forward Pass
    try:
        img = torch.randn(1, 3, 224, 224)
        out = model(img)
        print(f"✅ Forward pass successful. Output length: {len(out)}")
        print(f"   - Feature shape: {out[0].shape}")
        print(f"   - Code shape: {out[1].shape}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return

    print("All checks passed! BiomedCLIP is connected and functioning properly.")

if __name__ == "__main__":
    main()
