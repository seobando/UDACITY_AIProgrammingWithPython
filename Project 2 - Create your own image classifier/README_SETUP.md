# Image Classifier Project - Setup Instructions

This guide will help you set up the virtual environment for the Image Classifier project.

## Prerequisites

- Python 3.8 or higher
- pip (usually comes with Python)

## Quick Setup

### Option 1: Using PowerShell Script (Recommended for Windows)

1. Open PowerShell in the project directory
2. Run:
   ```powershell
   .\setup_venv.ps1
   ```

If you get an execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Option 2: Using Batch Script (Windows CMD)

1. Open Command Prompt in the project directory
2. Run:
   ```cmd
   setup_venv.bat
   ```

### Option 3: Manual Setup

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   ```

2. **Activate virtual environment:**
   
   On Windows PowerShell:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
   
   On Windows CMD:
   ```cmd
   venv\Scripts\activate.bat
   ```
   
   On Linux/Mac:
   ```bash
   source venv/bin/activate
   ```

3. **Upgrade pip:**
   ```bash
   python -m pip install --upgrade pip
   ```

4. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

## Verify Installation

After setup, verify that packages are installed:

```bash
python -c "import torch; import torchvision; print('PyTorch version:', torch.__version__)"
```

## Using the Virtual Environment

### Activate the environment:

**Windows PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows CMD:**
```cmd
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### Deactivate the environment:

```bash
deactivate
```

## Running the Project

### Training a model:

```bash
python image-classifier-part-1/train.py flowers --gpu
```

### Making predictions:

```bash
python image-classifier-part-1/predict.py path/to/image.jpg checkpoint.pth --gpu
```

## Troubleshooting

### Issue: "python: command not found"
- Make sure Python is installed and added to PATH
- Try using `python3` instead of `python`

### Issue: "Execution Policy" error in PowerShell
- Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Issue: "failed to locate pyvenv.cfg"
- This usually indicates a corrupted Python installation
- Try reinstalling Python or using a different Python installation

### Issue: CUDA/GPU not available
- Install CUDA-enabled PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- Or use CPU-only version (default)

## Requirements

The project requires:
- torch>=2.0.0
- torchvision>=0.15.0
- Pillow>=9.0.0
- numpy>=1.21.0
- matplotlib>=3.5.0
- requests>=2.28.0

All dependencies are listed in `requirements.txt`.

