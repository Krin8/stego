"""
STEGO Environment & Hardware Verification Script for Windows (Dell G15)
Run this script to verify GPU acceleration, CUDA availability, and all dependencies:
    python verify_windows_setup.py
"""

import sys
import platform
import os
from pathlib import Path

def print_header(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def check_system():
    print_header("System & Python Environment")
    print(f"OS: {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print(f"Machine: {platform.machine()} - Processor: {platform.processor()}")
    print(f"Python: {platform.python_version()} ({sys.executable})")
    
    if sys.version_info < (3, 10):
        print("⚠️ Warning: Python version < 3.10. Recommended: Python 3.10 or 3.11.")
    elif sys.version_info >= (3, 13):
        print("⚠️ Warning: Python version >= 3.13. Some PyTorch CUDA wheels may lack full stability.")
    else:
        print("✅ Python version is optimal (3.10 - 3.12).")

def check_cuda():
    print_header("NVIDIA GPU & CUDA Status (Dell G15)")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            vram_gb = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
            cuda_version = torch.version.cuda
            
            print(f"✅ CUDA is AVAILABLE!")
            print(f"   CUDA Runtime Version: {cuda_version}")
            print(f"   GPU Count: {gpu_count}")
            print(f"   Active Device: {gpu_name}")
            print(f"   Dedicated VRAM: {vram_gb:.2f} GB")
            
            # Quick tensor test on GPU
            x = torch.randn(100, 100, device="cuda")
            y = x @ x
            print("✅ Tensor computation on GPU succeeded!")
        else:
            print("❌ CUDA is NOT available. PyTorch is running in CPU mode.")
            print("   Check that:")
            print("   1. NVIDIA GeForce drivers are installed on your Dell G15.")
            print("   2. You installed PyTorch with CUDA (e.g. cu121 index).")
    except ImportError:
        print("❌ PyTorch is not installed.")

def check_dependencies():
    print_header("Package Import Diagnostics")
    
    packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("pytorch_lightning", "PyTorch Lightning"),
        ("hydra", "Hydra Core"),
        ("omegaconf", "OmegaConf"),
        ("pydicom", "PyDicom"),
        ("SimpleITK", "SimpleITK"),
        ("cv2", "OpenCV"),
        ("open_clip", "OpenCLIP"),
        ("timm", "Timm"),
        ("transformers", "HuggingFace Transformers"),
        ("scipy", "SciPy"),
        ("sklearn", "Scikit-Learn"),
        ("skimage", "Scikit-Image"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("PIL", "Pillow"),
        ("tqdm", "TQDM"),
        ("kornia", "Kornia"),
        ("einops", "Einops"),
        ("tensorboard", "TensorBoard")
    ]
    
    all_passed = True
    for module_name, display_name in packages:
        try:
            mod = __import__(module_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  ✅ {display_name:<25} : {ver}")
        except ImportError as e:
            print(f"  ❌ {display_name:<25} : NOT FOUND ({e})")
            all_passed = False
            
    # Test opencv-contrib specific feature used in crf.py
    try:
        import cv2
        _ = cv2.ximgproc.jointBilateralFilter
        print("  ✅ OpenCV ximgproc (CRF)     : Available")
    except AttributeError:
        print("  ⚠️ OpenCV ximgproc (CRF)     : Missing! Install opencv-contrib-python instead of opencv-python")
        all_passed = False
    except ImportError:
        pass
        
    return all_passed

def check_biomedclip_weights():
    print_header("BiomedCLIP Model Weights Check")
    weights_path = Path(__file__).parent / "src" / "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224" / "open_clip_pytorch_model.bin"
    config_path = Path(__file__).parent / "src" / "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224" / "open_clip_config.json"
    
    if weights_path.exists() and config_path.exists():
        size_mb = weights_path.stat().st_size / (1024**2)
        print(f"✅ BiomedCLIP local weights found ({size_mb:.1f} MB) at: {weights_path}")
    else:
        print(f"⚠️ BiomedCLIP local weights missing at: {weights_path}")
        print("   Run: python -c \"from huggingface_hub import snapshot_download; snapshot_download('microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', local_dir='src/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')\"")

if __name__ == "__main__":
    check_system()
    check_cuda()
    ok = check_dependencies()
    check_biomedclip_weights()
    
    print("\n" + "=" * 60)
    if ok:
        print("🎉 ENVIRONMENT READY: All core packages and components verified!")
    else:
        print("⚠️ Some dependencies are missing. Run the installation script or check requirements.")
    print("=" * 60 + "\n")
