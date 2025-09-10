@echo off
REM AIFX Trading Demo Launcher for Windows
REM ======================================

echo 🚀 Starting AIFX Trading Demo...

REM Set Python path for module imports
set PYTHONPATH=src\main\python

REM Check if jsonschema is installed
echo 🔍 Checking dependencies...
python -c "import jsonschema; print('✅ jsonschema available')" 2>nul
if %errorlevel% neq 0 (
    echo ❌ jsonschema not found - installing...
    pip install jsonschema
)

REM Check TensorFlow status
python -c "import tensorflow as tf; print('✅ TensorFlow available:', tf.__version__)" 2>nul
if %errorlevel% neq 0 (
    echo ⚠️  TensorFlow not available - LSTM model will be disabled
)

REM Launch the trading demo
echo 🎯 Launching AIFX Trading Demo...
python run_trading_demo.py --mode demo

pause