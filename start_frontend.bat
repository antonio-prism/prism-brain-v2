@echo off
REM ============================================
REM PRISM Brain v2 - Start Frontend Dashboard
REM ============================================
REM This starts the Streamlit dashboard
REM Make sure the backend is running first!
REM ============================================

echo ============================================
echo PRISM Brain v2 - Starting Frontend Dashboard
echo ============================================

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo Virtual environment activated.
) else (
    echo WARNING: No virtual environment found.
    echo Create one first with: py -3.11 -m venv venv
    echo Then install deps: pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo Starting Streamlit dashboard...
echo The dashboard will open in your browser automatically.
echo Press Ctrl+C to stop
echo.

cd frontend
streamlit run Welcome.py

pause
