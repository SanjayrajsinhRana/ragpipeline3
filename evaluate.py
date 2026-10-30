#!/usr/bin/env python3
"""
Evaluation for the iMac Environmental Report RAG.

Retrieval: does a retrieved chunk contain every required fact? (rank, MRR)
Answer:    does the generated answer contain every required fact? Refusal
           questions must answer with the exact refusal sentence.

Each fact is a list of accepted spellings (any one must appear).
Run:  python evaluate.py [--rerank] [--no-generate] [--compare]
"""

import argparse
import re
import time

from src.generation import REFUSAL
from src.pipeline import IMacRagPipeline

CASES = [
    # --- the three required sample questions
    {"id": "S1", "kind": "text/footnote", "q": "How much less energy does iMac use compared to the ENERGY STAR requirement?",
     "facts": [["58 percent", "58%"]]},
    {"id": "S2", "kind": "table+chart", "q": "What is the total carbon footprint of the iMac, and how much of that comes from product use versus manufacturing?",
     "facts": [["346 kg"], ["45%", "45 percent"], ["50%", "50 percent"]]},
    {"id": "S3", "kind": "callout", "q": "What material is the iMac stand made from, and what percentage of it is recycled?",
     "facts": [["aluminum"], ["100 percent", "100%"]]},
    # --- chart
    {"id": "C1", "kind": "chart", "q": "What was the baseline emissions figure, and how many kilograms did the reductions remove?",
     "facts": [["445"], ["99"]]},
    {"id": "C2", "kind": "chart", "q": "Why are renewable energy emissions not visible on the emissions chart?",
     "facts": [["too small"]]},
    # --- tables
    {"id": "T1", "kind": "table", "q": "How much power does the iMac draw in Idle with the display on at 115V?",
     "facts": [["20.6"]]},
    {"id": "T2", "kind": "table", "q": "What is the power adapter efficiency at 230V?",
     "facts": [["91.9"]]},
    {"id": "T3", "kind": "table", "q": "What is the carbon footprint of the iMac with four ports and 512GB?",
     "facts": [["391"]]},
    # --- footnotes
    {"id": "F1", "kind": "footnote", "q": "How is recycled content defined for the iMac?",
     "facts": [["mass of certified recycled material"], ["packaging"]]},
    {"id": "F2", "kind": "footnote", "q": "Which rating did iMac achieve on the EPEAT registry?",
     "facts": [["gold"]]},
    {"id": "F3", "kind": "footnote", "q": "What does the Idle display-on power mode mean in the ENERGY STAR power consumption table?",
     "facts": [["auto-brightness", "brightness"]]},
    # --- text
    {"id": "X1", "kind": "text", "q": "How much recycled content does the iMac packaging contain?",
     "facts": [["76"]]},
    # --- refusals (must not hallucinate)
    {"id": "R1", "kind": "refusal", "q": "How much does the iMac cost?", "refusal": True},
    {"id": "R2", "kind": "refusal", "q": "What is the battery capacity of the iMac?", "refusal": True},
]


def norm(s):
    return re.sub(r"\s+", " ", s.lower().replace("co 2 e", "co2e"))


def has_facts(text, facts):
    t = norm(text)
    return all(any(norm(alt) in t for alt in group) for group in facts)


def eval_retrieval(rag):
    print(f"\n=== RETRIEVAL (top_k={rag.top_k}, reranker={'on' if rag.use_reranker else 'off'}) ===")
    ranks, mrr, n = {}, 0.0, 0
    for c in CASES:
        if c.get("refusal"):
            continue
        n += 1
        hits = rag.retrieve(c["q"])
        # facts may be split over chunks, so accept the best single chunk first
        rank = next((i for i, h in enumerate(hits, 1) if has_facts(h["chunk"]["text"], c["facts"])), None)
        joint = rank is None and has_facts(" ".join(h["chunk"]["text"] for h in hits), c["facts"])
        ranks[c["id"]] = rank
        mrr += 1 / rank if rank else 0
        label = f"rank {rank}" if rank else ("PASS (facts spread over chunks)" if joint else "FAIL")
        top = hits[(rank or 1) - 1]["chunk"]
        print(f"[{c['id']}] {c['kind']:<13} {label:<8} <- p.{top['page_start']} {top['kind']}")
    found = sum(1 for r in ranks.values() if r)
    print(f"Retrieval: {found}/{n} single-chunk hits | MRR {mrr / n:.3f}")
    return mrr / n


def eval_answers(rag):
    print("\n=== END-TO-END ANSWERS (Gemini) ===")
    ok = 0
    for c in CASES:
        res = rag.query(c["q"])
        ans = res["answer"]
        passed = REFUSAL.lower() in ans.lower() if c.get("refusal") else has_facts(ans, c["facts"])
        ok += passed
        print(f"[{c['id']}] {'PASS' if passed else 'FAIL'}  {c['q']}\n      A: {ans}")
        time.sleep(1)
    print(f"\nAnswers: {ok}/{len(CASES)} passed")
    return ok == len(CASES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--no-generate", action="store_true")
    ap.add_argument("--compare", action="store_true", help="retrieval only: reranker off vs on")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    if a.compare:
        for rr in (False, True):
            eval_retrieval(IMacRagPipeline(use_reranker=rr, rebuild=a.rebuild and not rr))
        return
    rag = IMacRagPipeline(use_reranker=a.rerank, rebuild=a.rebuild)
    eval_retrieval(rag)
    if not a.no_generate:
        eval_answers(rag)


if __name__ == "__main__":
    main()
