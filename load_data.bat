@echo off
REM ============================================
REM PRISM Brain v2 - Load Data into Database
REM ============================================
REM Run this ONCE after setting up PostgreSQL
REM to populate the database with 174 risk events
REM ============================================

echo ============================================
echo PRISM Brain v2 - Database Data Loader
echo ============================================

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo Virtual environment activated.
) else (
    echo WARNING: No virtual environment found.
    pause
    exit /b 1
)

echo.
echo Step 1: Loading 174 risk events into PostgreSQL...
echo.
python load_events.py

echo.
echo Step 2: Regenerating indicator weights...
echo.
python regenerate_weights.py

echo.
echo ============================================
echo DATA LOADING COMPLETE!
echo ============================================
echo.
echo Next: Run start_backend.bat and start_frontend.bat
echo.

pause
