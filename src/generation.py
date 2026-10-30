"""
Grounded answer generation with Gemini (free tier), temperature 0.
Answers only from numbered passages, cites [n], refuses otherwise.
"""

import os
import time

from dotenv import load_dotenv

load_dotenv()

REFUSAL = "I could not find this in the document."

SYSTEM_PROMPT = f"""You answer questions about one document using ONLY the numbered context passages.
- Use only facts stated in the passages. Never add outside knowledge.
- Passages come from body text, tables ("Table: ... Columns: ..." with one row per line), chart descriptions and footnotes. Read a table value from the row whose label matches the question and the column that matches it.
- Quote numbers, units and names exactly as written.
- You may add or subtract figures that the passages give, but say that the result is calculated.
- If a passage has a footnote that limits or defines the claim, mention it briefly.
- Cite the passages you used with square brackets only, like [1] or [2][3]. Plain text, no bold.
- If the passages do not contain the answer, reply exactly: {REFUSAL}
- Be brief: one to three sentences."""


def build_context(hits):
    return "\n\n".join(f"[{i}] {h['chunk']['text']}" for i, h in enumerate(hits, 1))


def answer_question(question, hits, model=None, retries=4):
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set (put it in .env). Free key: https://aistudio.google.com/")
    model = os.environ.get("GEMINI_MODEL") or model or "gemini-3.1-flash-lite"
    client = genai.Client(api_key=api_key)
    prompt = f"Context passages:\n\n{build_context(hits)}\n\nQuestion: {question}"
    for attempt in range(retries):
        try:
            r = client.models.generate_content(
                model=model, contents=prompt,
                config={"system_instruction": SYSTEM_PROMPT, "temperature": 0.0})
            return (r.text or REFUSAL).strip()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(8 * (attempt + 1))
