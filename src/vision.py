"""
Chart / diagram / icon descriptions from Gemini vision (free tier).

Text extraction keeps a chart's labels but not what they add up to (e.g. the
bars "445 kg", "-99 kg" and "346 kg"). One vision call per visual page turns
that into searchable prose. Results are cached on disk, so a page is only sent
once per (page image, prompt).
"""

import hashlib
import json
import os
import time

import pymupdf
from dotenv import load_dotenv

load_dotenv()

NO_VISUAL = "NONE"

PROMPT = """You are helping index a product report for question answering.
This image is one page of the report. Describe ONLY the information that is
carried by non-paragraph content: charts, graphs, diagrams, tables drawn as
graphics, icon call-outs and large statistic call-outs.

Rules:
- Copy every printed number, unit, label and footnote marker exactly (a small
  raised number after a label is a footnote marker; write it as [note N]).
- For a chart: say what kind it is, what each bar/segment/axis is, the printed
  value of each bar, and how the values relate (for example "A minus B equals C"
  when the page shows that).
- Never estimate a value that is not printed. If a segment has no printed
  number, say it is not labelled with a value.
- Do not repeat ordinary body paragraphs.
- If the page has no chart, diagram or call-out content beyond paragraphs, reply
  with exactly: NONE
Reply in plain sentences, no markdown, at most 200 words."""


def _key(page_no, png):
    """Cache key: page image + prompt. The model is deliberately not part of it,
    so a free-tier daily quota running out on one model never forces a re-run."""
    h = hashlib.sha256()
    h.update(PROMPT.encode())
    h.update(png)
    return f"{page_no}:{h.hexdigest()[:16]}"


def render_page(pdf_path, page_no, dpi=130):
    doc = pymupdf.open(pdf_path)
    return doc[page_no - 1].get_pixmap(dpi=dpi).tobytes("png")


def describe_pages(pdf_path, page_numbers, cache_path, model=None, verbose=True):
    """Return {page_no: description or None}. Uses and updates the JSON cache."""
    from google import genai
    from google.genai import types

    model = model or os.environ.get("GEMINI_VISION_MODEL") or os.environ.get("GEMINI_MODEL")
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    client = genai.Client(api_key=api_key) if api_key else None
    out = {}
    for page_no in page_numbers:
        png = render_page(pdf_path, page_no)
        key = _key(page_no, png)
        if key not in cache:
            if client is None:
                out[page_no] = None  # no key: skip, do not fail ingestion
                continue
            cache[key] = _call(client, types, model, png)
            with open(cache_path, "w") as f:
                json.dump(cache, f, indent=1)
            if verbose:
                print(f"  vision: page {page_no} described ({len(cache[key])} chars)")
        text = cache[key].strip()
        out[page_no] = None if text.upper().startswith(NO_VISUAL) else text
    return out


def _call(client, types, model, png, retries=5):
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=png, mime_type="image/png"), PROMPT],
                config={"temperature": 0.0},
            )
            return resp.text or NO_VISUAL
        except Exception as e:  # free tier: 429 / transient 5xx
            if attempt == retries - 1:
                raise
            time.sleep(10 * (attempt + 1))
