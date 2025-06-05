"""FastAPI server for HuggingFace model inference."""

import logging
import random
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .config import settings

classifier = None
security = HTTPBearer()

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger("uvicorn")

MICHAL_FACTS = [
    "Michal loves board games and always wins at strategy games!",
    "Michal enjoys hiking and discovering new mountain trails!",
    "Michal is a coffee enthusiast who codes best with good espresso!",
    "Michal loves sci-fi movies and can quote Star Wars perfectly!",
    "Michal enjoys table tennis and never loses a match at work!",
    "Michal loves teambuildings and always brings positive energy!",
    "Michal is passionate about new technologies and AI innovations!",
    "Michal enjoys PC games and board games equally!",
    "Michal loves good food and discovering new restaurants!",
    "Michal is great at go-karts and laser games!",
]


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str


class PredictResponse(BaseModel):
    """Prediction response model."""

    prediction: dict


class PredictRequest(BaseModel):
    """Prediction request model."""

    text: str


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan."""
    global classifier  # noqa: PLW0603
    try:
        logger.info("Loading model: %s", settings.MODEL_NAME)

        tokenizer = AutoTokenizer.from_pretrained(
            settings.MODEL_NAME,
            cache_dir=settings.CACHE_DIR,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            settings.MODEL_NAME,
            cache_dir=settings.CACHE_DIR,
        )

        classifier = pipeline("text-generation", model=model, tokenizer=tokenizer)

        logger.info("Model loaded successfully")
    except Exception as e:
        logger.exception("Error loading model")
        error_msg = f"Failed to load model: {e!s}"
        raise RuntimeError(error_msg) from e

    yield
    logger.info("Application shutting down...")


app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    lifespan=lifespan,
)

_security_dependency = Depends(security)


def authenticate(
    credentials: HTTPAuthorizationCredentials = _security_dependency,
) -> bool:
    """Authenticate user with Bearer token."""
    if credentials.credentials != settings.API_TOKEN:
        logger.warning(
            "Invalid authentication attempt: %s...",
            credentials.credentials[:10],
        )
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing authentication token",
        )
    return True


@app.get("/health")
async def health_check() -> HealthResponse:
    """Check server status."""
    return HealthResponse(status="OK")


@app.middleware("http")
async def log_requests(request: Request, call_next: Callable) -> Response:
    """Log HTTP requests and responses."""
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"

    logger.info(
        "Request: %s %s from %s",
        request.method,
        request.url.path,
        client_ip,
    )

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info("Response: %s - time: %.4fs", response.status_code, process_time)

    return response


@app.post(
    "/predict",
    dependencies=[Depends(authenticate)],
)
async def predict(request: PredictRequest) -> PredictResponse:
    """Generate prediction with humor enhancement."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        answer_prompt = f"Q: {request.text}\nA:"

        logger.info("Executing prediction for text: %s...", request.text[:50])

        answer_prediction = classifier(
            answer_prompt,
            max_new_tokens=25,
            do_sample=True,
            temperature=0.7,
            truncation=True,
        )

        answer = (
            answer_prediction[0]["generated_text"].replace(answer_prompt, "").strip()
        )
        michal_fact = random.choice(MICHAL_FACTS)  # noqa: S311

        logger.info("Michal fact: %s", michal_fact)

        return PredictResponse(
            prediction={
                "question": request.text,
                "answer": answer,
                "michal_fact": michal_fact,
            },
        )

    except Exception as e:
        logger.exception("Error during prediction")
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )
