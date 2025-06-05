from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import logging
from contextlib import asynccontextmanager
from .config import settings
from pydantic import BaseModel

classifier = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier
    try:
        logger.info(f"Loading model: {settings.MODEL_NAME}")

        tokenizer = AutoTokenizer.from_pretrained(
            settings.MODEL_NAME, cache_dir=settings.CACHE_DIR
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            settings.MODEL_NAME, cache_dir=settings.CACHE_DIR
        )

        classifier = pipeline("text-generation", model=model, tokenizer=tokenizer)

        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")

    yield
    logger.info("Application shutting down...")


app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    lifespan=lifespan,
)

security = HTTPBearer()

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger("uvicorn")


class HealthResponse(BaseModel):
    status: str


# Authenticate
def authenticate(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != settings.API_TOKEN:
        logger.warning(
            f"Not valid try for authenticate with token: {credentials.credentials[:10]}..."
        )
        raise HTTPException(
            status_code=401, detail="Not valid or missing authenticate token"
        )
    return True


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check status of server.
    """
    return HealthResponse(status="OK")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )
