# PowerShell script to set up virtual environment for Image Classifier Project

Write-Host "Setting up virtual environment for Image Classifier Project..." -ForegroundColor Green

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: Python not found. Please install Python 3.8 or higher." -ForegroundColor Red
    exit 1
}

# Remove existing venv if it exists
if (Test-Path "venv") {
    Write-Host "Removing existing venv..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force venv
}

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Green
python -m venv venv

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error creating virtual environment. Trying alternative method..." -ForegroundColor Yellow
    # Try with python3
    python3 -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Could not create virtual environment." -ForegroundColor Red
        Write-Host "Please ensure Python is properly installed and try again." -ForegroundColor Red
        exit 1
    }
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& "venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Green
python -m pip install --upgrade pip

# Install requirements
Write-Host "Installing requirements..." -ForegroundColor Green
pip install -r requirements.txt

Write-Host "`nSetup complete! To activate the virtual environment, run:" -ForegroundColor Green
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "`nOr on Windows CMD:" -ForegroundColor Green
Write-Host "  venv\Scripts\activate.bat" -ForegroundColor Cyan

