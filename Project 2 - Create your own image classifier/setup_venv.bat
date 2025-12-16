@echo off
REM Batch script to set up virtual environment for Image Classifier Project

echo Setting up virtual environment for Image Classifier Project...

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.8 or higher.
    exit /b 1
)

REM Remove existing venv if it exists
if exist venv (
    echo Removing existing venv...
    rmdir /s /q venv
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Error: Could not create virtual environment.
    echo Please ensure Python is properly installed and try again.
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

echo.
echo Setup complete! To activate the virtual environment, run:
echo   venv\Scripts\activate.bat
echo.

pause

