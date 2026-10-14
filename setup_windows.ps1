# STEGO Windows Setup Script for PowerShell (Dell G15)
$ErrorActionPreference = "Stop"

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  STEGO Windows Setup for Dell G15 Laptop" -ForegroundColor Cyan
Write-Host "  Configuring PyTorch with CUDA & All Dependencies" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

$hasConda = Get-Command conda -ErrorAction SilentlyContinue

if ($hasConda) {
    Write-Host "[INFO] Conda detected. Creating/updating 'stego' environment..." -ForegroundColor Green
    conda env create -f environment-windows.yml
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[INFO] Environment may exist, running update..." -ForegroundColor Yellow
        conda env update -f environment-windows.yml --prune
    }
    Write-Host "`nTo activate run: conda activate stego" -ForegroundColor Green
    & conda run -n stego python verify_windows_setup.py
} else {
    Write-Host "[INFO] Conda not found. Using standard Python virtual environment..." -ForegroundColor Yellow
    $hasPython = Get-Command python -ErrorAction SilentlyContinue
    if (-not $hasPython) {
        Write-Host "[ERROR] Python is not installed or not in PATH." -ForegroundColor Red
        Write-Host "Please install Python 3.11 from https://www.python.org/downloads/ (Check 'Add to PATH')." -ForegroundColor Red
        exit 1
    }

    if (-not (Test-Path ".venv")) {
        Write-Host "[STEP 1/4] Creating virtual environment (.venv)..." -ForegroundColor Cyan
        python -m venv .venv
    }

    Write-Host "[STEP 2/4] Activating .venv and upgrading pip..." -ForegroundColor Cyan
    & .\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel

    Write-Host "[STEP 3/4] Installing PyTorch with CUDA 12.1 for NVIDIA RTX GPU..." -ForegroundColor Cyan
    & .\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    Write-Host "[STEP 4/4] Installing project requirements..." -ForegroundColor Cyan
    & .\.venv\Scripts\python.exe -m pip install -r requirements-windows.txt

    Write-Host "`n[VERIFY] Running diagnostics..." -ForegroundColor Cyan
    & .\.venv\Scripts\python.exe verify_windows_setup.py
}

Write-Host "`n================================================================" -ForegroundColor Green
Write-Host "  Setup completed!" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
