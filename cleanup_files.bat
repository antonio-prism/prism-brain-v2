@echo off
echo ========================================
echo PRISM Brain V2 - File Cleanup Script
echo ========================================
echo.
echo This script removes files that were already removed from git
echo but still exist physically on your PC.
echo.
pause

echo Removing dead frontend code...
del /Q "frontend\api_client.py" 2>nul
del /Q "frontend\modules\demo_data.py" 2>nul
del /Q "frontend\modules\smart_prioritization.py" 2>nul
del /Q "frontend\data\risk_events_v2.json" 2>nul
del /Q "frontend\BUGFIX_NOTES.md" 2>nul
del /Q "frontend\README.md" 2>nul
echo Done.

echo Removing Phase 3 disabled pages...
rmdir /S /Q "frontend\pages_disabled" 2>nul
echo Done.

echo Removing Phase 3 backend routes...
del /Q "dashboard_routes.py" 2>nul
echo Done.

echo.
echo ========================================
echo Cleanup complete!
echo ========================================
echo.
echo Next: Run the app to verify everything works:
echo   1. Open Git Bash
echo   2. cd ~/Documents/prism-brain-v2
echo   3. source venv/Scripts/activate
echo   4. python -m uvicorn main:app --host 0.0.0.0 --port 8000
echo.
pause
