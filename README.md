# iMac Product Environmental Report — Visual-Document RAG

Question answering over a short, brochure-style PDF (12 pages) where the key
facts live in a **chart, tables, icon call-outs and footnotes**, not in
paragraphs. Free tier only: local HuggingFace models + Gemini free-tier key.

## Pipeline

```
PDF ─► ingest.py ─► element extraction ─┬─ text blocks (column-aware, footnote markers inline)
                                        ├─ tables rebuilt under dark header bars
                                        ├─ endnotes parsed from the "Endnotes" pages
                                        └─ vision.py: Gemini describes chart/diagram pages (cached)
     ─► chunking.py: ONE chunk per table / call-out / paragraph / chart / endnote,
                     each claim carries its footnote text
     ─► embeddings.py: BAAI/bge-small-en-v1.5 (local) + BM25
     ─► retrieval.py: RRF (k=60) over both lists  (reranker optional, off)
     ─► generation.py: grounded Gemini prompt, cites [n], exact refusal
```

| Problem in this document | Technique |
|---|---|
| Table value separated from its column header | Find dark header bars in the page drawings; rebuild rows from word positions → `Idle—Display On: 100V = 20.5W; 115V = 20.6W; ...` |
| Chart numbers (445 kg, −99 kg, 346 kg) lose their relationship | One Gemini vision call per page → prose like "445 kg minus 99 kg equals 346 kg" |
| Footnote marker becomes a stray digit; caveat on another page | Detect superscript spans (`flags & 1`) → `[note 18]`, parse endnotes, append `Footnotes: [18] ...` to the claim's chunk; also index each endnote with the claim that cites it |
| Cards / columns interleave | Lines join an element only if left-aligned (or centred) and directly below |
| Big stat + label split ("100%" / "of the packaging") | Lead-stat merge |
| Tables cut by size | One chunk per element, never split by character count |

Reranker (`ms-marco-MiniLM`) is **off by default**: trained on prose, it can
under-rate table/chart chunks. `python evaluate.py --compare` measures it
(here: MRR 0.903 off vs 0.917 on — within noise, so the simpler path is kept).

## Setup

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
cp .env.example .env        # add GEMINI_API_KEY (free: https://aistudio.google.com/)
./venv/bin/python main.py --rebuild "What material is the iMac stand made from?"
```

Free-tier Gemini quotas are **per model per day** (some allow only ~20 requests),
so `GEMINI_MODEL` (answers) and `GEMINI_VISION_MODEL` (chart descriptions) are
separate. Vision results are cached in `data/vision_cache.json` (key = page
image + prompt, not model), so the index can be rebuilt without new API calls.
Run scripts without `-I` (they import `src/`).

## Usage

```bash
./venv/bin/python main.py "How much less energy does iMac use than ENERGY STAR requires?"
./venv/bin/python main.py                  # interactive
./venv/bin/python main.py --rebuild        # re-ingest + re-embed
./venv/bin/python evaluate.py              # retrieval + end-to-end answers
./venv/bin/python evaluate.py --compare    # reranker off vs on
```

## Results (14 hand-checked questions: 3 required samples, chart, table, footnote, text, 2 refusals)

* Retrieval: 12/12 answer-bearing chunks in top 6, MRR 0.90
* Answers: **14/14 pass**, including both out-of-scope refusals
  (answer facts must appear in the text; refusal must be exact)

Required samples: *58 percent less energy* · *346 kg CO2e, production 50% / product use 45%* ·
*aluminum, 100% recycled*.

## Notes / limits

* Table detection is tuned for this document's style (dark header bars); the idea transfers, thresholds may not.
* The report states 346 kg for "iMac (two ports) 256GB" in the table but footnote 10 says "512GB storage". The pipeline reports what each passage says.
* Vision text is model-generated; prompt forbids estimating unlabeled values.
