# Windows Setup & Run Commands (Dell G15)

Quick copy-paste commands for setting up and running **STEGO** on Windows (Dell G15 with NVIDIA RTX GPU).

---

## Method 1: Using Conda (Recommended)

Open **Anaconda Prompt** or **PowerShell**:

```powershell
# 1. Clone repository
git clone https://github.com/Krin8/stego.git
cd stego

# 2. Create Conda environment (Python 3.11 + PyTorch CUDA 12.1)
conda env create -f environment-windows.yml

# 3. Activate environment
conda activate stego

# 4. Download BiomedCLIP model weights
python -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', local_dir='src/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')"

# 5. Verify GPU, CUDA, and packages
python verify_windows_setup.py
```

---

## Method 2: Using Python venv + pip

Open **Command Prompt (cmd)** or **PowerShell**:

```cmd
:: 1. Clone repository
git clone https://github.com/Krin8/stego.git
cd stego

:: 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

:: 3. Upgrade pip and build tooling
python -m pip install --upgrade pip setuptools wheel

:: 4. Install PyTorch with CUDA 12.1 (NVIDIA RTX GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

:: 5. Install all dependencies
pip install -r requirements-windows.txt

:: 6. Download BiomedCLIP weights
python -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', local_dir='src/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')"

:: 7. Verify GPU, CUDA, and packages
python verify_windows_setup.py
```

---

## Method 3: One-Click Setup Script

In File Explorer inside the `stego` folder:
- Double click **`setup_windows.bat`**  
*(or in PowerShell run `.\setup_windows.ps1`)*

---

## Verification & Testing Commands

```powershell
# Verify full environment and CUDA status
python verify_windows_setup.py

# Test BiomedCLIP model connection
python src/test_biomedclip.py

# Download standard STEGO pretrained checkpoints (DINO, MoCo, PiCIE)
python src/download_models.py
```

---

## Running Pipelines: src (src 1) vs src2 (src 2)

### 1. `src` (Primary PyTorch Lightning Pipeline)
```powershell
# View available training arguments and configs
python src/train_segmentation.py --help

# Test data loader and preprocessing
python src/preprocess.py --help
```

### 2. `src2` (Optimized P1 Pipeline with Pinned Checkpoint)
```powershell
# Run smoke tests for P1 backbone
python src2/tests/smoke_backbone.py

# Run full numerical and pipeline tests
python src2/tests/test_pipeline.py

# Test P1 kNN neighbors
python src2/tests/check_knn_neighbors.py
```

---

## Useful Dell G15 Tips

| Action | Command / Shortcut |
| :--- | :--- |
| **Activate G-Mode (High Fan / Boost)** | Press `F9` or `Fn + F9` |
| **Check GPU Status in Terminal** | `nvidia-smi` |
| **If out of VRAM (RTX 3050 4GB/6GB)** | Add `batch_size=4` or `batch_size=2` to training command |
