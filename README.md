# PRISM Brain Probability Engine v2

Risk probability calculation engine for 900 risk events.

## Quick Deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/prism-brain)

### Manual Deployment

1. Fork this repository
2. Create a new project on [Railway](https://railway.app)
3. Add a **PostgreSQL** database service
4. Add a **Redis** service
5. Deploy from your GitHub repo
6. Railway will automatically detect and deploy

### API Endpoints

Once deployed, visit your app URL to see:
- `/health` - Health check
- `/docs` - Interactive API documentation
- `/api/v1/events` - List risk events
- `/api/v1/probabilities` - View calculations

## Local Development

```bash
docker-compose up -d postgres redis
pip install -r requirements.txt
python scripts/railway_init.py
uvicorn api.main:app --reload
```

## Documentation

See the `/docs` folder for detailed documentation.
