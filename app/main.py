from fastapi import FastAPI
from fastapi.security import HTTPBearer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import logging
from contextlib import asynccontextmanager
from .config import settings

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )
