from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name, device="cpu", batch_size=64):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)

    def encode_unique(self, texts):
        unique = list(dict.fromkeys(texts))
        if not unique:
            return {}, np.empty((0, 0), dtype=np.float32)

        emb = self.model.encode(
            unique,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        lookup = {text: i for i, text in enumerate(unique)}
        return lookup, emb

    @staticmethod
    def cosine(a, b):
        # Embeddings are normalized, so dot product = cosine similarity.
        return float(np.dot(a, b))
