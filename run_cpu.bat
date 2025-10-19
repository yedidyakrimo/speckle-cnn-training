@echo off
echo Starting Speckle CNN Training with CPU...
echo ========================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Run training with CPU
echo Starting training with CPU configuration...
python run_training.py --cpu

echo.
echo Training completed!
pause
