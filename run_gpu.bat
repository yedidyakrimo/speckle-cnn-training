@echo off
echo Starting Speckle CNN Training with GPU...
echo ========================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check GPU availability
echo Checking GPU configuration...
python check_gpu.py
echo.

REM Run training with GPU
echo Starting training with GPU configuration...
python run_training.py --gpu

echo.
echo Training completed!
pause
