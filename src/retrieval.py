"""
Hybrid retrieval: dense cosine + BM25, merged with Reciprocal Rank Fusion
(k=60). An optional cross-encoder reranker is available but OFF by default:
it is trained on prose and under-rates table / chart chunks (measured in
evaluate.py).
"""

import numpy as np
from sentence_transformers import CrossEncoder

from src.embeddings import tokenize_for_bm25

RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_ce = None


def _top(scores, k):
    idx = np.argsort(scores)[::-1][:k]
    return [int(i) for i in idx]


def hybrid_search(index, query, pool=20, rrf_k=60):
    dense = _top(index.embeddings @ index.encode_query(query), pool)
    sparse = _top(index.bm25.get_scores(tokenize_for_bm25(query)), pool)
    fused = {}
    for ranking in (dense, sparse):
        for rank, i in enumerate(ranking, start=1):
            fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + rank)
    return [{"chunk": index.chunks[i], "score": s, "index": i}
            for i, s in sorted(fused.items(), key=lambda kv: -kv[1])]


def rerank(query, cands, top_n):
    global _ce
    _ce = _ce or CrossEncoder(RERANKER)
    scores = _ce.predict([(query, c["chunk"]["text"]) for c in cands])
    ranked = sorted(zip(cands, scores), key=lambda t: -t[1])
    return [{**c, "score": float(s)} for c, s in ranked[:top_n]]


def retrieve(index, query, top_k=6, use_reranker=False, pool=20, rrf_k=60):
    cands = hybrid_search(index, query, pool, rrf_k)
    return rerank(query, cands, top_k) if use_reranker else cands[:top_k]
