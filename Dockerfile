FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./requirements.txt
COPY frontend/requirements.txt ./frontend_requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r frontend_requirements.txt

# Copy application code
COPY . .

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501

# Start both services
CMD ["bash", "start_app.sh"]
