import logging
import threading
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from app.config import Config

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Thread-safe singleton service for generating vector embeddings using Sentence Transformers.
    Caches model weights in memory with lazy loading and supports batched, normalized encoding.
    """
    _instance: Optional["EmbeddingService"] = None
    _lock = threading.Lock()

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or Config.EMBEDDING_MODEL_NAME
        self.dimension = Config.EMBEDDING_DIM
        self._model: Optional[SentenceTransformer] = None
        self._load_lock = threading.Lock()

    @property
    def model(self) -> SentenceTransformer:
        """
        Lazily loads the model into memory on the first actual embedding call.
        """
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    logger.info(f"Loading embedding model '{self.model_name}' into memory...")
                    # Explicitly use token=False unless HF_TOKEN is provided to prevent
                    # huggingface_hub from sending expired OAuth tokens found in ~/.cache/huggingface/token
                    import os
                    hf_token = os.getenv("HF_TOKEN") or False
                    self._model = SentenceTransformer(self.model_name, token=hf_token)
                    self.dimension = self._model.get_sentence_embedding_dimension()
                    logger.info(
                        f"Embedding model loaded. Vector dimension: {self.dimension}"
                    )
        return self._model

    @classmethod
    def get_instance(cls, model_name: Optional[str] = None) -> "EmbeddingService":
        """
        Thread-safe singleton accessor.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(model_name)
        return cls._instance

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generates normalized vector embeddings for a list of text strings.

        :param texts: List of chunk text strings.
        :param batch_size: Batch size for forward pass.
        :return: List of float vectors (each vector is 384 dimensions).
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()
        return [list(e) if hasattr(e, "__iter__") else e for e in embeddings]

    def embed_query(self, query: str) -> List[float]:
        """
        Generates an embedding vector for a single search query string.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty")

        result = self.embed_texts([query.strip()], batch_size=1)
        return result[0]
