# HuggingFace API Server

REST API server for HuggingFace model inference with humor enhancement featuring facts about Michal.

## Overview

This FastAPI application provides a REST API interface for text generation using Microsoft's DialoGPT model from HuggingFace. The server includes Bearer token authentication, request logging, and adds fun facts about Michal to each prediction response.

## Requirements

- **Python 3.10+**
- **Docker & Docker Compose**
- **4GB+ RAM** (for model loading)
- **Internet connection** (for initial model download)

## Installation & Setup

### 1. Clone the Repository

- git clone `https://github.com/Michal-Barcak/HuggingFace-API-Server.git`
- cd `HuggingFace-API-Server`


### 2. Environment Configuration

- Create `.env` file and edit with your own configuration:

    ```
    MODEL_NAME=microsoft/DialoGPT-large
    API_TOKEN=your-secret-token-here
    HOST=0.0.0.0
    PORT=8000
    LOG_LEVEL=INFO
    CACHE_DIR=./model_cache
    ```

### 3. Start with Docker (Recommended)

#### Build and start the application
- `docker-compose up --build -d`

#### View logs
- `docker-compose logs -f`

#### Stop the application
- `docker-compose down`

## API Documentation

### Base URL
- `http://localhost:8000`

### Authentication
All protected endpoints require Bearer token authentication:
- Authorization: Bearer your-secret-token


### Endpoints

#### 1. Health Check
Check if the server is running.

**Request:**
```
GET /health
```

**Response:**
```
{
"status": "OK"
}
```

#### 2. Text Prediction
Generate text prediction with humor enhancement.

**Request:**
```
POST /predict
Content-Type: application/json
Authorization: Bearer your-secret-token

{
"text": "What is artificial intelligence?"
}
```

**Response:**
```
{
"prediction": {
"question": "What is artificial intelligence?",
"answer": "Artificial intelligence is a field of computer science that aims to create intelligent machines...",
"michal_fact": "Michal is passionate about new technologies and AI innovations!"
}
}
```


## Testing Examples

### Using curl
- Health check
- `curl http://localhost:8000/health`

- Prediction request
```python
curl -X POST http://localhost:8000/predict \
-H "Authorization: Bearer your-secret-token" \
-H "Content-Type: application/json" \
-d "{\"text\": \"Tell me about machine learning\"}"
```

### Interactive Documentation

Visit `http://localhost:8000/docs` for Swagger UI with interactive API documentation.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `microsoft/DialoGPT-large` | HuggingFace model identifier |
| `API_TOKEN` | `default-dev-token` | Bearer token for authentication |
| `HOST` | `127.0.0.1` | Server host address |
| `PORT` | `8000` | Server port |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `CACHE_DIR` | `./model_cache` | Directory for model caching |

### Model Options

You can use different DialoGPT models by changing `MODEL_NAME` for example:
- `microsoft/DialoGPT-small` (117M parameters) - Faster, less memory
- `microsoft/DialoGPT-medium` (345M parameters) - Balanced
- `microsoft/DialoGPT-large` (762M parameters) - Best quality, more memory

## Development

### Code Quality

#### Format code with Black
- `black app/`

#### Lint code with Ruff
- `ruff check app/`

#### Fix auto-fixable issues
- `ruff check --fix app/`


### Dependencies

Main dependencies include:
- **FastAPI** - Web framework
- **Uvicorn** - ASGI server
- **Transformers** - HuggingFace models
- **PyTorch** - Deep learning framework
- **Pydantic** - Data validation
- **python-dotenv** - Environment variables

## Monitoring & Logging

The application includes comprehensive logging:

    INFO: Request: POST /predict from 172.18.0.1
    INFO: Executing prediction for text: What is artificial intelligence?...
    INFO: Michal fact: Michal loves board games and always wins at strategy games!
    INFO: Response: 200 - time: 2.3456s


## 🚨 Troubleshooting

### Common Issues

**Model Loading Fails:**
- Check internet connection
- Verify sufficient disk space (2GB+)
- Ensure `CACHE_DIR` has write permissions

**Authentication Errors:**
```
    {
    "detail": "Invalid or missing authentication token"
    }
```
- Verify `API_TOKEN` in `.env` file
- Include correct `Authorization: Bearer <token>` header

**Slow Response Times:**
- First request takes longer (model loading)
- Subsequent requests are faster (model cached)
- Consider using smaller model for development

**Docker Issues:**

- Rebuild if needed
- `docker-compose down`
- `docker-compose up --build -d`

- Check logs
- `docker-compose logs -f hf-api-server`