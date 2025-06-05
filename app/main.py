from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import logging
from contextlib import asynccontextmanager
from .config import settings
from pydantic import BaseModel
import time
import random

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


class PredictResponse(BaseModel):
    prediction: dict


class PredictRequest(BaseModel):
    text: str


def authenticate(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != settings.API_TOKEN:
        logger.warning(
            f"Not valid try for authenticate with token: {credentials.credentials[:10]}..."
        )
        raise HTTPException(
            status_code=401, detail="Not valid or missing authenticate token"
        )
    return True


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check status of server.
    """
    return HealthResponse(status="OK")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"

    logger.info(f"Request: {request.method} {request.url.path} od {client_ip}")

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(f"Answer: {response.status_code} - čas: {process_time:.4f}s")

    return response


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, authorized: bool = Depends(authenticate)):
    try:
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")

        if not request.text or request.text.strip() == "":
            raise HTTPException(status_code=400, detail="Text could not be empty")

        answer_prompt = f"Q: {request.text}\nA:"

        logger.info(f"Executed prediction for text: {request.text[:50]}...")
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

        michal_facts = [
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

        michal_fact = random.choice(michal_facts)
        logger.info(f"Michal fact: {michal_fact}")

        return PredictResponse(
            prediction={
                "question": request.text,
                "answer": answer,
                "michal_fact": michal_fact,
            }
        )

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )
