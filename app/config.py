import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    MODEL_NAME: str = os.getenv("MODEL_NAME", "microsoft/DialoGPT-small")
    CACHE_DIR: str = os.getenv("CACHE_DIR", "./model_cache")

    API_TOKEN: str = os.getenv("API_TOKEN", "default-dev-token")

    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    API_TITLE: str = "HuggingFace API Server"
    API_DESCRIPTION: str = (
        "REST API server for HuggingFace model inference with humor enhancement"
    )
    API_VERSION: str = "1.0.0"


settings = Settings()
