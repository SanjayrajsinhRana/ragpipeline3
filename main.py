#!/usr/bin/env python3
"""Ask questions about the iMac Product Environmental Report.

    python main.py "What material is the iMac stand made from?"
    python main.py            # interactive
    python main.py --rebuild  # re-ingest, re-embed (vision descriptions stay cached)
"""

import argparse

from src.pipeline import IMacRagPipeline


def show(res, verbose):
    print(f"\nA: {res['answer']}")
    if verbose:
        print("\nSources:")
        for i, h in enumerate(res["hits"], 1):
            c = h["chunk"]
            print(f"  [{i}] p.{c['page_start']} {c['kind']:<12} {c['section'][:40]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--rerank", action="store_true", help="enable cross-encoder reranker (off by default)")
    ap.add_argument("-q", "--quiet", action="store_true", help="hide sources")
    a = ap.parse_args()
    rag = IMacRagPipeline(rebuild=a.rebuild, use_reranker=a.rerank)
    if a.question:
        show(rag.query(a.question), not a.quiet)
        return
    print("Ask about the iMac report (empty line to quit).")
    while True:
        q = input("\nQ: ").strip()
        if not q:
            break
        show(rag.query(q), not a.quiet)


if __name__ == "__main__":
    main()
