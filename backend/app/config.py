import os
from pathlib import Path
from dotenv import load_dotenv

# Locate and load the .env file from project root or backend directory
# Looks for .env in current folder or 2 levels up (documind/.env)
current_dir = Path(__file__).resolve().parent
for parent in [current_dir.parent, current_dir.parent.parent]:
    candidate = parent / ".env"
    if candidate.exists():
        load_dotenv(dotenv_path=candidate)
        break


class Config:
    # Database configuration
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql://documind:documind@localhost:5432/documind"
    )

    # File upload configurations
    UPLOAD_FOLDER = Path(__file__).resolve().parent.parent / "uploads"
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload limit
    ALLOWED_EXTENSIONS = {"pdf", "docx"}

    # Text Chunking configurations
    # Tuned for all-MiniLM-L6-v2 (256 token limit):
    # 500 characters is approx 100-120 words (130-160 tokens)
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

    # Embedding configurations
    EMBEDDING_MODEL_NAME = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    EMBEDDING_DIM = 384

    # Reranking and Hybrid Search configurations
    RERANKER_MODEL_NAME = os.getenv(
        "RERANKER_MODEL_NAME",
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    RRF_K = int(os.getenv("RRF_K", "60"))

    # Google Gemini LLM configurations
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))  # Low temperature for factual precision
    LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1024"))
    LLM_REQUEST_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", "60"))
