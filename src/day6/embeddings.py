"""Local embedding model via sentence-transformers (no API key, cost 0)."""
from pathlib import Path
from typing import Protocol

from huggingface_hub import try_to_load_from_cache


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class SentenceTransformerEmbedder:
    """E5-family model: documents get "passage: ", queries get "query: ".

    Vectors are L2-normalized so cosine distance in pgvector ranks correctly.
    Loading errors propagate: there is no fallback to mock vectors.
    """

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        self.dimensions = self._model.get_embedding_dimension()
        # Snapshot hash of the downloaded model files, for reproducibility.
        config_path = try_to_load_from_cache(model_name, "config.json")
        self.revision = Path(config_path).parent.name if isinstance(config_path, str) else None

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                     show_progress_bar=False)
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode([f"passage: {text}" for text in texts])

    def embed_query(self, question: str) -> list[float]:
        return self._encode([f"query: {question}"])[0]
