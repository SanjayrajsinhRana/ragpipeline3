"""
Element-level chunking: one chunk per table, per call-out/paragraph, per chart
description and per endnote. Nothing is cut by character count, so a table row
never gets separated from its header and a claim stays with its footnote.

Every chunk is text of the form

    [Page 7 | Use | text]
    iMac uses 58 percent less energy ... [note 18]
    Footnotes: [18] Energy consumption and energy efficiency values ...

The header line carries page, section and element type into both the dense
embedding and BM25.
"""

import re

MAX_WORDS = 220      # long paragraphs are split on sentence boundaries
MIN_WORDS = 4        # shorter fragments are chart labels / captions, not facts
LEAD_STAT_RE = re.compile(r"\d")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[“\"])")
ABBREV = ("U.S.", "No.", "Inc.", "e.g.", "i.e.", "vs.", "approx.", "Stat.")


def split_sentences(text):
    protected = text
    for a in ABBREV:
        protected = protected.replace(a, a.replace(".", "․"))
    return [s.replace("․", ".") for s in SENTENCE_RE.split(protected)]


def pack_sentences(text, max_words=MAX_WORDS):
    """Split a long paragraph into whole-sentence pieces."""
    if len(text.split()) <= max_words:
        return [text]
    pieces, cur = [], []
    for s in split_sentences(text):
        if cur and len(" ".join(cur + [s]).split()) > max_words:
            pieces.append(" ".join(cur))
            cur = []
        cur.append(s)
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def section_for(el, headings, page_title):
    """Nearest heading above the element in its own column, else page title."""
    above = [h for h in headings
             if h["y0"] <= el["y0"] + 2 and abs(h["x0"] - el["x0"]) <= 60 and h is not el
             and not LEAD_STAT_RE.search(h["text"])]
    if above:
        return max(above, key=lambda h: h["y0"])["text"]
    return page_title


def merge_lead_stats(elements):
    """Attach a big number heading ("100%", "445 kg") to the label under it."""
    out, skip = [], set()
    for i, el in enumerate(elements):
        if i in skip:
            continue
        if el["heading"] and LEAD_STAT_RE.search(el["text"]) and len(el["text"].split()) <= 4:
            nxt = next((j for j, o in enumerate(elements)
                        if j != i and j not in skip and not o["heading"]
                        and 0 <= o["y0"] - el["y1"] <= 16
                        and abs((o["x0"] + o["x1"]) / 2 - (el["x0"] + el["x1"]) / 2) <= 60), None)
            if nxt is not None:
                o = elements[nxt]
                skip.add(nxt)
                out.append({**o, "text": f"{el['text']} {o['text']}", "y0": el["y0"],
                            "markers": el.get("markers", []) + o.get("markers", []),
                            "heading": False})
                continue
        out.append(el)
    return out


def build_chunks(doc, visual):
    """doc: ingest_pdf() result; visual: {page_no: description or None}."""
    notes = doc["endnotes"]
    chunks = []

    def add(page, section, kind, body, markers=()):
        foot = []
        for m in dict.fromkeys(markers):  # unique, ordered
            n = int(m)
            if n in notes:
                foot.append(f"[{n}] {notes[n]['text']}")
        text = f"[Page {page} | {section} | {kind}]\n{body}"
        if foot:
            text += "\nFootnotes: " + " ".join(foot)
        chunks.append({"text": text, "page_start": page, "page_end": page,
                       "kind": kind, "section": section, "body": body})

    cited_by = {}  # note number -> (page, claim) for endnote chunks

    for page in doc["pages"]:
        p = page["page"]
        els = merge_lead_stats(page["elements"])
        headings = [e for e in els if e["heading"]]
        page_title = max(headings, key=lambda h: h["size"])["text"] if headings else f"Page {p}"

        for el in els:
            words = len(el["text"].split())
            sec = section_for(el, headings, page_title)
            is_fact_heading = el["heading"] and words >= 5
            if el["heading"] and not is_fact_heading:
                continue                       # pure section label: context only
            if words < MIN_WORDS and not el.get("markers"):
                continue                       # stray chart label
            if words < MIN_WORDS:
                continue
            for piece in pack_sentences(el["text"]):
                add(p, sec, "text", piece, re.findall(r"\[note (\d+)\]", piece))
            for m in re.findall(r"\[note (\d+)\]", el["text"]):
                cited_by.setdefault(int(m), (p, re.sub(r"\s*\[note \d+\]", "", el["text"])[:240]))

        for t in page["tables"]:
            markers = re.findall(r"\[note (\d+)\]", t["text"])
            add(p, t["title"], "table", t["text"], markers)
            for m in markers:
                cited_by.setdefault(int(m), (p, t["title"]))

        desc = visual.get(p)
        if desc:
            add(p, page_title, "chart/visual", desc, re.findall(r"\[note (\d+)\]", desc))
            for m in re.findall(r"\[note (\d+)\]", desc):
                cited_by.setdefault(int(m), (p, "chart"))

    for n, note in sorted(notes.items()):
        page, claim = cited_by.get(n, (None, None))
        where = f" cited on page {page} by: \"{claim}\"" if page else ""
        chunks.append({
            "text": f"[Page {note['page']} | Endnote {n} | endnote]\nEndnote {n}{where}\n{note['text']}",
            "page_start": note["page"], "page_end": note["page"], "kind": "endnote",
            "section": f"Endnote {n}", "body": note["text"]})
    return chunks
