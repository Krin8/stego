@echo off
setlocal enabledelayedexpansion

echo ================================================================
echo   STEGO Windows Setup for Dell G15 Laptop
echo   Configuring PyTorch with CUDA & All Dependencies
echo ================================================================

:: Check for Python or Conda
where conda >nul 2>nul
if %errorlevel% equ 0 (
    echo [INFO] Conda found. Using Conda for optimal Windows setup...
    echo [STEP 1/3] Creating Conda environment 'stego' from environment-windows.yml...
    call conda env create -f environment-windows.yml
    if %errorlevel% neq 0 (
        echo [WARNING] Conda env create had an issue, trying to update existing env...
        call conda env update -f environment-windows.yml --prune
    )
    echo [STEP 2/3] Activating 'stego' environment...
    call conda activate stego
) else (
    echo [INFO] Conda not found. Falling back to Python venv...
    where python >nul 2>nul
    if %errorlevel% neq 0 (
        echo [ERROR] Python is not found in PATH!
        echo Please install Python 3.11 from https://www.python.org/downloads/
        echo Make sure to check "Add Python to PATH" during installation.
        pause
        exit /b 1
    )
    
    echo [STEP 1/4] Creating virtual environment (.venv)...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    
    echo [STEP 2/4] Upgrading pip and wheel...
    python -m pip install --upgrade pip setuptools wheel
    
    echo [STEP 3/4] Installing PyTorch with CUDA 12.1 for NVIDIA RTX GPU...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    echo [STEP 4/4] Installing project requirements...
    pip install -r requirements-windows.txt
)

echo.
echo ================================================================
echo   Running Hardware & Dependency Verification...
echo ================================================================
python verify_windows_setup.py

echo.
echo ================================================================
echo   Setup finished! 
echo   To activate this environment later, run:
echo     conda activate stego    (if using Conda)
echo     .venv\Scripts\activate  (if using venv)
echo ================================================================
pause
