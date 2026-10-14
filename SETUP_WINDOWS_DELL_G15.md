# Complete Windows Setup Guide for Dell G15 (NVIDIA RTX)

This guide provides step-by-step instructions to set up the **STEGO** project on your Windows laptop (**Dell G15** with NVIDIA GeForce RTX GPU).

---

## 1. Laptop & Hardware Optimization (Dell G15)

The Dell G15 series is equipped with dedicated NVIDIA GeForce RTX graphics (RTX 3050, 3060, 4050, 4060, or similar) alongside an integrated Intel/AMD GPU.

### Prerequisites to Check Before Installing
1. **NVIDIA GeForce Drivers**:
   - Update your GPU driver to the latest Game Ready or Studio Driver via **GeForce Experience** or [nvidia.com/drivers](https://www.nvidia.com/download/index.aspx).
2. **NVIDIA Control Panel Setting**:
   - Open **NVIDIA Control Panel** → **Manage 3D settings** → **Preferred graphics processor**: Select **High-performance NVIDIA processor**.
3. **Dell Game Shift / G-Mode** (Optional but Recommended for Training):
   - Press `F9` (or `Fn + F9`) on your Dell G15 keyboard to activate **G-Mode** (Dynamic Boost & high fan profile) during heavy model training.
4. **Microsoft Visual C++ Redistributable**:
   - Download and install the latest **Visual C++ 2015–2022 Redistributable (x64)** from Microsoft:  
     🔗 [Direct Microsoft Download](https://aka.ms/vs/17/release/vc_redist.x64.exe)  
     *(Required by PyTorch, OpenCV, and SciPy compiled binaries).*
5. **Enable Long Paths in Windows (Recommended)**:
   - Open PowerShell as Administrator and run:
     ```powershell
     New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
       -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
     ```

---

## 2. Fast Setup (Automated)

We provide one-click setup scripts in the project root:

### Option A: Using Command Prompt (`.bat`)
Double-click `setup_windows.bat` or run in CMD:
```cmd
setup_windows.bat
```

### Option B: Using PowerShell (`.ps1`)
Right-click `setup_windows.ps1` → **Run with PowerShell**, or in a PowerShell terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_windows.ps1
```

---

## 3. Manual Step-by-Step Setup

If you prefer doing it manually, follow either **Method 1 (Conda - Recommended)** or **Method 2 (Python venv)**.

### Method 1: Using Miniconda / Anaconda (Recommended)

Conda handles CUDA runtime libraries and binary dependencies cleanly on Windows.

1. **Download & Install Miniconda** (if not already installed):
   - [Miniconda Windows 64-bit Installer](https://docs.anaconda.com/miniconda/)
2. **Open Anaconda Prompt** (or PowerShell with conda initialized).
3. **Navigate to the repository folder**:
   ```cmd
   cd path\to\STEGO
   ```
4. **Create the environment from `environment-windows.yml`**:
   ```cmd
   conda env create -f environment-windows.yml
   ```
5. **Activate the environment**:
   ```cmd
   conda activate stego
   ```

---

### Method 2: Using Standard Python (venv + pip)

If you use official Python from [python.org](https://www.python.org/downloads/):

1. **Install Python 3.11 (64-bit)**:
   - ⚠️ **Important**: On the installer screen, check the box:  
     ☑ **"Add Python to PATH"**
2. **Open Command Prompt or PowerShell** in the `STEGO` root folder:
   ```cmd
   cd path\to\STEGO
   ```
3. **Create and activate a virtual environment**:
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```
4. **Upgrade pip and build tooling**:
   ```cmd
   python -m pip install --upgrade pip setuptools wheel
   ```
5. **Install PyTorch with CUDA 12.1 for NVIDIA RTX GPU**:
   ```cmd
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
6. **Install all remaining dependencies**:
   ```cmd
   pip install -r requirements-windows.txt
   ```

---

## 4. Download Pretrained Weights & Assets

### A. BiomedCLIP Weights
The BiomedCLIP backbone weights are expected in `src/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/`. If the large `.bin` file is missing, download it from Hugging Face:

```cmd
python -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', local_dir='src/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')"
```

### B. Standard STEGO Pretrained Weights (DINO / MoCo / PiCIE)
Run the built-in downloader:
```cmd
python src/download_models.py
```

---

## 5. Verify the Setup

Run the included verification script to test GPU acceleration and all packages:

```cmd
python verify_windows_setup.py
```

Expected output:
```text
============================================================
 NVIDIA GPU & CUDA Status (Dell G15)
============================================================
PyTorch version: 2.x.x+cu121
✅ CUDA is AVAILABLE!
   Active Device: NVIDIA GeForce RTX ... Laptop GPU
   Dedicated VRAM: ... GB
✅ Tensor computation on GPU succeeded!

============================================================
 Package Import Diagnostics
============================================================
  ✅ PyTorch                   : 2.x.x+cu121
  ✅ TorchVision               : ...
  ✅ PyTorch Lightning         : ...
  ✅ Hydra Core                : ...
  ✅ OmegaConf                 : ...
  ✅ PyDicom                   : ...
  ✅ OpenCV                    : ...
  ✅ OpenCV ximgproc (CRF)     : Available
  ...
🎉 ENVIRONMENT READY: All core packages and components verified!
```

You can also run the quick BiomedCLIP backbone connection test:
```cmd
python src/test_biomedclip.py
```

---

## 6. Windows-Specific Notes & Troubleshooting

### 1. `ValueError: Invalid sharing strategy: 'file_system'`
- **Reason**: PyTorch on Windows does not support Unix IPC `file_system` sharing strategy.
- **Status**: **Already patched** in `src/train_segmentation.py` and `src/demo_segmentation.py` with `if sys.platform != 'win32':`.

### 2. OpenCV `AttributeError: module 'cv2' has no attribute 'ximgproc'`
- **Reason**: Standard `opencv-python` omits extra modules needed by Dense CRF (`jointBilateralFilter`).
- **Solution**: Always use `opencv-contrib-python` (already configured in `requirements-windows.txt`).

### 3. DataLoader Multiprocessing on Windows
- If you encounter memory issues or slow worker startup in PyTorch on Windows, set `num_workers=0` or `num_workers=2` in your training config. On Windows, Python uses `spawn` instead of `fork`, which has slightly higher per-worker memory overhead.

### 4. GPU VRAM Considerations (4GB - 8GB on Dell G15)
- If your Dell G15 has an RTX 3050 (4GB/6GB) or RTX 4050 (6GB), reduce the batch size if you encounter `CUDA out of memory`:
  ```bash
  python src/train_segmentation.py batch_size=4
  ```
