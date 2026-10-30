"""
Dense (BAAI/bge-small-en-v1.5, local) + sparse (BM25) index over the chunks,
pickled to disk so a restart does not recompute embeddings.
"""

import pickle
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*%?")


def tokenize_for_bm25(text):
    """Lowercase tokens that keep numbers like 20.6 and 58% whole."""
    return _TOKEN_RE.findall(text.lower())


class DocumentIndex:
    def __init__(self, chunks, embedder=None):
        self.chunks = chunks
        self.embedder = embedder or SentenceTransformer(EMBED_MODEL)
        self.embeddings = self.embedder.encode(
            [c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=False)
        self.bm25 = BM25Okapi([tokenize_for_bm25(c["text"]) for c in chunks])

    def encode_query(self, q):
        return self.embedder.encode([QUERY_PREFIX + q], normalize_embeddings=True)[0]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "embeddings": self.embeddings}, f)

    @classmethod
    def load(cls, path, embedder=None):
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.chunks = data["chunks"]
        obj.embeddings = data["embeddings"]
        obj.embedder = embedder or SentenceTransformer(EMBED_MODEL)
        obj.bm25 = BM25Okapi([tokenize_for_bm25(c["text"]) for c in obj.chunks])
        return obj
