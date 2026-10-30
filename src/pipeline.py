"""
End-to-end RAG pipeline for the iMac Product Environmental Report.

  ingest (text + tables + endnotes) -> vision (charts) -> element chunks
  -> bge-small + BM25 index -> RRF retrieval -> grounded Gemini answer
"""

import os
import time

from src.chunking import build_chunks
from src.embeddings import DocumentIndex
from src.generation import answer_question
from src.ingest import ingest_pdf
from src.retrieval import retrieve
from src.vision import describe_pages


class IMacRagPipeline:
    def __init__(self, pdf_path="data/iMac_PER_Oct2024.pdf",
                 index_path="data/index_cache.pkl",
                 vision_cache="data/vision_cache.json",
                 top_k=6, use_reranker=False, rebuild=False, use_vision=True):
        self.pdf_path, self.top_k, self.use_reranker = pdf_path, top_k, use_reranker
        if os.path.exists(index_path) and not rebuild:
            self.index = DocumentIndex.load(index_path)
        else:
            self.index = self.build(index_path, vision_cache, use_vision)

    def build(self, index_path, vision_cache, use_vision):
        print("Ingesting PDF (text, tables, endnotes)...")
        doc = ingest_pdf(self.pdf_path)
        visual = {}
        if use_vision:
            pages = [p["page"] for p in doc["pages"] if not p["is_endnotes"]]
            print(f"Describing visuals on {len(pages)} pages with Gemini vision (cached)...")
            visual = describe_pages(self.pdf_path, pages, vision_cache)
        chunks = build_chunks(doc, visual)
        print(f"Built {len(chunks)} chunks; embedding...")
        index = DocumentIndex(chunks)
        index.save(index_path)
        return index

    def retrieve(self, question):
        return retrieve(self.index, question, top_k=self.top_k, use_reranker=self.use_reranker)

    def query(self, question):
        t0 = time.time()
        hits = self.retrieve(question)
        t1 = time.time()
        answer = answer_question(question, hits)
        return {"question": question, "answer": answer, "hits": hits,
                "retrieval_time": t1 - t0, "generation_time": time.time() - t1,
                "total_time": time.time() - t0}
