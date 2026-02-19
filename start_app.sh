#!/bin/bash
# PRISM Brain V2 - Railway Deployment Startup Script
# Starts both FastAPI backend and Streamlit frontend

echo "=== PRISM Brain V2 Starting ==="
echo "PORT: ${PORT:-8000}"

# Start FastAPI backend in background
echo "Starting FastAPI backend on port ${PORT:-8000}..."
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start Streamlit frontend on port 8501
echo "Starting Streamlit frontend on port 8501..."
cd frontend
streamlit run Welcome.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false &
FRONTEND_PID=$!

echo "=== Both services started ==="
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"

# Wait for either process to exit
wait -n $BACKEND_PID $FRONTEND_PID

# If one dies, kill the other
echo "=== A service exited, shutting down ==="
kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
wait
