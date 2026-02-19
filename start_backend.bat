@echo off
REM ============================================
REM PRISM Brain v2 - Start Backend API
REM ============================================
REM This starts the FastAPI backend on port 8000
REM Make sure PostgreSQL is running first!
REM ============================================

echo ============================================
echo PRISM Brain v2 - Starting Backend API
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

REM Check if .env exists
if not exist ".env" (
    echo WARNING: .env file not found!
    echo Copy .env.example or create one with DATABASE_URL
    pause
    exit /b 1
)

echo.
echo Starting FastAPI server on http://localhost:8000
echo Press Ctrl+C to stop
echo.

python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

pause
