from __future__ import annotations


class SentenceTransformerEmbedder:
    """Pinned sentence-transformers model, CPU by default (used only to rank label similarity)."""

    def __init__(self, model_id: str, revision: str, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer  # heavy import kept local

        self._model_id = model_id
        self._revision = revision
        self._model = SentenceTransformer(model_id, revision=revision, device=device)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, v)) for v in vectors]

    def describe(self) -> dict[str, str]:
        return {"model_id": self._model_id, "revision": self._revision}
