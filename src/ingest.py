"""
Element-aware PDF ingestion for short, visual documents.

Instead of one flat text stream, each page is split into *elements*, each of
which becomes its own searchable unit:

  text      narrative blocks / icon call-outs (footnote markers kept inline)
  table     rebuilt from word positions under dark header bars, one row per
            line as "row label | column = value | ..."
  chart     page-level description of charts/diagrams from a vision model
            (see vision.py), because their numbers are not in the text layer
  endnote   numbered notes parsed from the "Endnotes" pages, each carrying the
            claim that cites it

Footnote markers are found via the PDF superscript font flag, so "Apple.7"
becomes "Apple. [note 7]" instead of a stray digit glued to a word.
"""

import re
import statistics
from collections import defaultdict

import pymupdf

SUPERSCRIPT = 1  # bit 0 of span flags
BOLD = 16
HEADER_BAR_MAX_LUMA = 0.45  # dark filled rectangles = table header bars
MIN_BAR_WIDTH = 60
ROW_Y_TOL = 3.0
CELL_GAP = 14.0
LINE_SPLIT_GAP = 15.0  # wider than any word space: separates side-by-side columns
ENDNOTE_RE = re.compile(r"^(\d{1,2})\s+(\S.*)$")
FOOTER_RE = re.compile(r"(Product Environmental Report|©\s*20\d\d Apple|^\d{1,2}$)")


def page_words(page):
    """Words with exact bboxes, split at superscript boundaries.

    Returns dicts: text, bbox, sup (bool), size, bold.
    """
    words = []
    raw = page.get_text("rawdict")
    for block in raw["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                sup = bool(span["flags"] & SUPERSCRIPT)
                cur, cur_box = "", None
                for ch in span["chars"]:
                    c = ch["c"]
                    if c.isspace():
                        if cur:
                            words.append(_word(cur, cur_box, sup, span))
                        cur, cur_box = "", None
                        continue
                    box = pymupdf.Rect(ch["bbox"])
                    cur += c
                    cur_box = box if cur_box is None else cur_box | box
                if cur:
                    words.append(_word(cur, cur_box, sup, span))
    return words


def _word(text, box, sup, span):
    return {
        "text": text,
        "bbox": tuple(box),
        "sup": sup and text.isdigit(),
        "size": span["size"],
        "bold": bool(span["flags"] & BOLD),
    }


def cluster_rows(words, tol=ROW_Y_TOL):
    """Group words into visual rows by vertical centre."""
    rows = []
    for w in sorted(words, key=lambda w: ((w["bbox"][1] + w["bbox"][3]) / 2, w["bbox"][0])):
        yc = (w["bbox"][1] + w["bbox"][3]) / 2
        if rows and abs(rows[-1]["y"] - yc) <= tol:
            rows[-1]["words"].append(w)
            rows[-1]["y"] = (rows[-1]["y"] + yc) / 2
        else:
            rows.append({"y": yc, "words": [w]})
    for r in rows:
        r["words"].sort(key=lambda w: w["bbox"][0])
    return rows


def row_to_cells(row_words, gap=CELL_GAP):
    """Split a row's words into cells wherever the horizontal gap is large."""
    cells, cur = [], []
    for w in row_words:
        if w["text"] in ("•", "·"):
            continue
        if cur and w["bbox"][0] - cur[-1]["bbox"][2] > gap:
            cells.append(cur)
            cur = []
        cur.append(w)
    if cur:
        cells.append(cur)
    return [_cell(c) for c in cells]


def _cell(ws):
    return {
        "text": join_words(ws),
        "x0": ws[0]["bbox"][0],
        "x1": ws[-1]["bbox"][2],
        "xc": (ws[0]["bbox"][0] + ws[-1]["bbox"][2]) / 2,
    }


def clean(text):
    """Whitespace + chemistry subscripts split by the font (CO 2 e -> CO2e)."""
    text = re.sub(r"\bCO\s*2\s*e\b", "CO2e", text)
    text = re.sub(r"(?<=[a-z])- (?=[a-z])", "-", text)  # "low- carbon" -> "low-carbon"
    return re.sub(r"\s+", " ", text).strip()


def join_words(ws):
    """Join words; superscript digits become inline [note n] markers.

    Words that touch the previous one (subscripts such as CO2e) are glued on.
    """
    out, prev = [], None
    for w in ws:
        if w["sup"]:
            out.append(f"[note {w['text']}]")
        elif prev is not None and not prev["sup"] and 0 <= w["bbox"][0] - prev["bbox"][2] < 1.2 \
                and w["bbox"][3] - w["bbox"][1] < (prev["bbox"][3] - prev["bbox"][1]) * 1.2:
            out[-1] += w["text"]
        else:
            out.append(w["text"])
        prev = w
    return " ".join(out).strip()


# ----------------------------------------------------------------- tables

def find_header_bars(page):
    """Dark filled rectangles wide enough to be table header cells."""
    bars = []
    for d in page.get_drawings():
        fill, r = d.get("fill"), d["rect"]
        if not fill or r.width < MIN_BAR_WIDTH or r.height > 60 or r.width > 560:
            continue
        luma = 0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2]
        if luma < HEADER_BAR_MAX_LUMA:
            bars.append(r)
    return bars


def group_bars(bars):
    """Merge adjacent bars that share a vertical band into one header."""
    groups = []
    for r in sorted(bars, key=lambda r: (r.y0, r.x0)):
        for g in groups:
            if abs(g["y0"] - r.y0) < 20 and r.x0 <= g["x1"] + 2:
                g["x1"] = max(g["x1"], r.x1)
                g["y1"] = max(g["y1"], r.y1)
                g["rects"].append(r)
                break
        else:
            groups.append({"x0": r.x0, "x1": r.x1, "y0": r.y0, "y1": r.y1, "rects": [r]})
    return groups


def extract_tables(page, words):
    """Rebuild tables that sit under dark header bars.

    Returns (tables, claimed_word_ids). Each table: dict(text, bbox, title).
    """
    groups = group_bars(find_header_bars(page))
    tables, claimed = [], set()
    for i, g in enumerate(groups):
        y_end = groups[i + 1]["y0"] - 2 if i + 1 < len(groups) else page.rect.y1 - 60
        in_region = [
            w for w in words
            if w["bbox"][1] >= g["y0"] - 2 and w["bbox"][3] <= y_end
            and w["bbox"][0] >= g["x0"] - 2 and w["bbox"][2] <= g["x1"] + 8
        ]
        table = _rebuild_table(cluster_rows(in_region), g)
        if table:
            claimed |= table.pop("word_ids")
            tables.append(table)
    return tables, claimed


def _header_columns(rows, g):
    """Return (label_header, anchors, title, header_row_count).

    Equal-height header cells side by side: one column per cell, header text
    is every word inside it (handles two-line headers like "iMac / 256GB").
    Otherwise (a title bar over a sub-header row): columns come from the
    last header row, earlier rows are the title.
    """
    rects = sorted(g["rects"], key=lambda r: r.x0)
    heights = {round(r.height) for r in rects}
    bar_rows = [r for r in rows if r["y"] <= g["y1"]]
    if len(rects) >= 2 and len(heights) == 1:
        cols = []
        for r in rects:
            ws = [w for row in bar_rows for w in row["words"]
                  if r.x0 - 1 <= (w["bbox"][0] + w["bbox"][2]) / 2 <= r.x1 + 1]
            if ws:
                cols.append({"text": " ".join(join_words(row["words"]) for row in cluster_rows(ws)),
                             "xc": (r.x0 + r.x1) / 2, "x0": r.x0})
        if len(cols) >= 2:
            return cols[0]["text"], cols[1:], None, len(bar_rows)
    if not bar_rows:
        return None
    anchors = row_to_cells(bar_rows[-1]["words"])
    title = " ".join(c["text"] for row in bar_rows[:-1] for c in row_to_cells(row["words"]))
    label = None
    if anchors and anchors[0]["x0"] <= g["x0"] + 6:
        label, anchors = anchors[0]["text"], anchors[1:]
    return (label, anchors, title or None, len(bar_rows)) if anchors else None


def _rebuild_table(rows, g):
    spec = _header_columns(rows, g)
    if not spec:
        return None
    label_header, anchors, title, n_header = spec
    first_value_x = anchors[0]["x0"] - 20

    lines, used, last_y, note = [], set(), None, None
    for r in [r for r in rows if r["y"] > g["y1"]]:
        if last_y is not None and r["y"] - last_y > 34:
            break
        cells = row_to_cells(r["words"])
        if len(cells) < 2:
            # a lone cell below the table body ends it; a "Note:" line belongs to it
            if len(cells) == 1 and cells[0]["text"].startswith("Note:"):
                note = cells[0]["text"]
                used |= {id(w) for w in r["words"]}
            break
        label = [c["text"] for c in cells if c["xc"] < first_value_x]
        vals = defaultdict(list)
        for c in cells:
            if c["xc"] >= first_value_x:
                a = min(anchors, key=lambda a: abs(a["xc"] - c["xc"]))
                vals[a["text"]].append(c["text"])
        parts = [f"{col} = {' '.join(v)}" for col, v in vals.items()]
        lines.append(f"{' '.join(label) or '(row)'}: " + "; ".join(parts))
        used |= {id(w) for w in r["words"]}
        last_y = r["y"]
    if not lines:
        return None

    used |= {id(w) for r in rows if r["y"] <= g["y1"] for w in r["words"]}
    col_names = ", ".join(a["text"] for a in anchors)
    name = " / ".join(x for x in (label_header, title) if x) or "Table"
    text = f"Table: {name}. Columns: {col_names}.\n" + "\n".join(lines)
    if note:
        text += "\n" + note
    return {"title": name, "text": clean_lines(text), "word_ids": used,
            "bbox": (g["x0"], g["y0"], g["x1"], (last_y or g["y1"]) + 8)}


def clean_lines(text):
    return "\n".join(clean(l) for l in text.split("\n"))


# ------------------------------------------------------------ text blocks

def page_lines(words):
    """Group words back into visual lines (rawdict order, y-clustered)."""
    lines = []
    for row in cluster_rows(words):
        ws = row["words"]
        # split a row into separate lines where the horizontal gap is huge
        seg = [ws[0]]
        for w in ws[1:]:
            if w["bbox"][0] - seg[-1]["bbox"][2] > LINE_SPLIT_GAP:
                lines.append(seg)
                seg = []
            seg.append(w)
        lines.append(seg)
    out = []
    for seg in lines:
        x0 = min(w["bbox"][0] for w in seg)
        x1 = max(w["bbox"][2] for w in seg)
        y0 = min(w["bbox"][1] for w in seg)
        y1 = max(w["bbox"][3] for w in seg)
        sizes = [w["size"] for w in seg if not w["sup"]] or [seg[0]["size"]]
        out.append({"words": seg, "x0": x0, "x1": x1, "y0": y0, "y1": y1,
                    "size": max(sizes), "bold": sum(w["bold"] for w in seg) / len(seg) > 0.6})
    return sorted(out, key=lambda l: (l["y0"], l["x0"]))


def extract_text_elements(words):
    """Column-aware text elements with inline footnote markers.

    A line joins an open element when it is left-aligned with it (or centred
    on it), directly below it, and the same font size. That keeps side-by-side
    cards and columns from interleaving.
    """
    if not words:
        return []
    body = statistics.median(w["size"] for w in words if not w["sup"])
    elements = []
    for ln in page_lines(words):
        best, best_gap = None, None
        for el in elements:
            gap = ln["y0"] - el["y1"]
            hanging = el["bullet"] and 0 <= ln["x0"] - el["x0"] <= 20
            aligned = hanging or abs(ln["x0"] - el["x0"]) <= 12 or \
                abs((ln["x0"] + ln["x1"]) / 2 - el["xc"]) <= 12
            if aligned and -2 <= gap <= 6 and abs(ln["size"] - el["size"]) < 1.5 \
                    and ln["bold"] == el["bold"]:
                if best is None or gap < best_gap:
                    best, best_gap = el, gap
        if best is None:
            elements.append({"lines": [ln], "x0": ln["x0"], "x1": ln["x1"], "y0": ln["y0"],
                             "y1": ln["y1"], "xc": (ln["x0"] + ln["x1"]) / 2,
                             "size": ln["size"], "bold": ln["bold"],
                             "bullet": ln["words"][0]["text"] in ("•", "·")})
        else:
            best["lines"].append(ln)
            best["y1"] = ln["y1"]
            best["x1"] = max(best["x1"], ln["x1"])
            best["xc"] = (best["x0"] + best["x1"]) / 2
    result = []
    for el in elements:
        ws = [w for ln in el["lines"] for w in ln["words"]]
        text = clean(" ".join(join_words(ln["words"]) for ln in el["lines"]))
        if not text or FOOTER_RE.search(text) and len(text) < 60:
            continue
        n_words = len(text.split())
        heading = n_words <= 14 and (el["size"] >= body * 1.25 or
                                     el["bold"] and n_words <= 8)
        result.append({"text": text, "heading": heading, "size": el["size"],
                       "x0": el["x0"], "x1": el["x1"], "y0": el["y0"], "y1": el["y1"],
                       "markers": [w["text"] for w in ws if w["sup"]]})
    return result


# --------------------------------------------------------------- endnotes

def parse_endnotes(doc, table_bboxes):
    """Parse numbered notes from pages headed 'Endnotes'.

    Note numbers sit in a left gutter (some are superscript glyphs) and run
    1, 2, 3 ... in order, so a row only starts a note when its first word is
    the next expected number. table_bboxes: {page_index: [bbox]} regions of
    tables that sit inside the endnotes pages and must not leak into a note.
    """
    notes, last = {}, 0
    for pno, page in enumerate(doc):
        if not re.search(r"^\s*Endnotes\s*$", page.get_text(), re.M):
            continue
        skip = [pymupdf.Rect(b) for b in table_bboxes.get(pno, [])]
        words = [w for w in page_words(page)
                 if not any(r.intersects(pymupdf.Rect(w["bbox"])) for r in skip)]
        cur = None
        for row in cluster_rows(words):
            line = join_words(row["words"])
            if line.startswith("©"):
                break
            if line == "Endnotes" or FOOTER_RE.search(line) and len(line) < 60:
                continue
            first = row["words"][0]
            if first["text"].isdigit() and int(first["text"]) == last + 1:
                last = cur = int(first["text"])
                body = " ".join(_plain(w) for w in row["words"][1:])
                notes[cur] = {"text": body, "page": pno + 1}
            elif cur is not None:
                body = join_words(row["words"])
                if body.startswith("•"):
                    notes[cur]["text"] += "\n- " + body.lstrip("• ").strip()
                else:
                    notes[cur]["text"] += " " + body
    for n in notes.values():
        n["text"] = clean_lines(n["text"])
    return notes


def _plain(w):
    return w["text"]


# ----------------------------------------------------------------- driver

def ingest_pdf(pdf_path):
    """Return {"pages": [...], "endnotes": {n: {text, page}}}."""
    doc = pymupdf.open(pdf_path)
    pages, table_bboxes = [], defaultdict(list)
    for pno, page in enumerate(doc):
        words = page_words(page)
        tables, claimed = extract_tables(page, words)
        for t in tables:
            table_bboxes[pno].append(t["bbox"])
        is_endnotes = bool(re.search(r"^\s*Endnotes\s*$", page.get_text(), re.M))
        free = [w for w in words if id(w) not in claimed]
        pages.append({
            "page": pno + 1,
            "elements": [] if is_endnotes else extract_text_elements(free),
            "tables": tables,
            "n_drawings": len(page.get_drawings()),
            "n_images": len(page.get_images()),
            "is_endnotes": is_endnotes,
        })
    return {"pages": pages, "endnotes": parse_endnotes(doc, table_bboxes)}
