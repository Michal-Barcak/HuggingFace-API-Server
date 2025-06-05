from fastapi import FastAPI
from fastapi.security import HTTPBearer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import logging
from contextlib import asynccontextmanager

classifier = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier
    try:
        logger.info("Loading model: microsoft/DialoGPT-large")

        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/DialoGPT-large", cache_dir="./model_cache"
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/DialoGPT-large", cache_dir="./model_cache"
        )

        classifier = pipeline("text-generation", model=model, tokenizer=tokenizer)

        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")

    yield
    logger.info("Application shutting down...")


app = FastAPI(
    title="HuggingFace API Server",
    description="REST API server for HuggingFace model inference with humor enhancement",
    version="1.0.0",
    lifespan=lifespan,
)

security = HTTPBearer()

logging.basicConfig(level=getattr(logging, "INFO"))
logger = logging.getLogger("uvicorn")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port="8000",
        reload=True,
        log_level="INFO",
    )
