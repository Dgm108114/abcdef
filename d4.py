import fitz
import re
import csv
import os

# =========================
# CONFIG
# =========================
PROS_ON_LEFT = True
PRINT_DEBUG = False

SECTIONS = [
    "Idea Generation",
    "Portfolio Construction",
    "Implementation",
    "Business Management",
]

SUBSECTION_HEADERS = [
    "Key takeaway",
    "Pros",
    "Cons",
]

BULLET_GLYPHS = ["•", "·", "◦", "●", "○"]

# Dynamic header/footer store
FORBIDDEN_DYNAMIC = set()


# =========================
# NORMALIZATION
# =========================
def normalize(s: str) -> str:
    if not s:
        return ""
    fixes = {
        "â€™": "’", "â€˜": "‘",
        "â€œ": "“", "â€�": "”",
        "â€“": "–", "â€”": "—",
        "â€¢": "•", "â€¦": "…",
        "Â": "",
    }
    for bad, good in fixes.items():
        s = s.replace(bad, good)
    s = s.replace("\u00A0", " ").replace("\u00AD", "")
    return re.sub(r"\s+", " ", s.strip())


# =========================
# DYNAMIC HEADER / FOOTER DETECTION
# =========================
def detect_repeating_header_footer_lines(doc):
    """
    Dynamically detect repeated header/footer lines using:
      - top/bottom 8% of page height
      - frequency across pages (>=2 occurrences)
    No hard-coded keywords.
    """
    header_candidates = []
    footer_candidates = []

    for page in doc:
        h = page.rect.height
        words = page.get_text("words") or []

        for w in words:
            t = normalize(w[4])
            if not t:
                continue
            y = w[1]

            # Top 8% = header band
            if y < 0.08 * h:
                header_candidates.append(t)

            # Bottom 8% = footer band
            elif y > 0.92 * h:
                footer_candidates.append(t)

    # Mark lines appearing on ≥2 pages
    header_footer = set()
    for group in (header_candidates, footer_candidates):
        freq = {}
        for t in group:
            freq[t] = freq.get(t, 0) + 1
        for t, c in freq.items():
            if c >= 2:
                header_footer.add(t)

    return header_footer


def is_header_footer(y, page_height, text):
    """
    Position + repetition + numeric page number logic.

    IMPORTANT:
    - We only treat dynamically detected header/footer lines (FORBIDDEN_DYNAMIC)
      as removable when they appear in header/footer zones.
    - This prevents company names that appear in headers from being removed
      when they appear again inside the main content.
    """
    if not text:
        return True

    t = normalize(text)

    # 1. Remove generic top/bottom bands
    if y < 0.08 * page_height:
        return True
    if y > 0.92 * page_height:
        # Numeric-only → definitely a page number
        if re.fullmatch(r"\d{1,4}", t):
            return True
        # All text in bottom band is treated as footer
        return True

    # 2. Remove repeated header/footer lines ONLY when in header/footer zones
    if t in FORBIDDEN_DYNAMIC:
        if y < 0.08 * page_height or y > 0.92 * page_height:
            return True

    return False


# =========================
# PAGE HELPERS
# =========================
def page_size(page):
    r = page.rect
    return r.width, r.height


def bottom_y(page):
    """Footer zone = bottom 8%."""
    h = page.rect.height
    return 0.92 * h


def group_words_into_lines(words, y_tol=3.0):
    if not words:
        return []
    ws = sorted(words, key=lambda w: (w[1], w[0]))
    lines = []
    current = [ws[0]]
    base_y = ws[0][1]

    for w in ws[1:]:
        if abs(w[1] - base_y) <= y_tol:
            current.append(w)
        else:
            lines.append(build_line(current))
            current = [w]
            base_y = w[1]

    if current:
        lines.append(build_line(current))
    return lines


def build_line(ws):
    ws_sorted = sorted(ws, key=lambda w: w[0])
    txt = " ".join(normalize(w[4]) for w in ws_sorted if w[4])
    return {
        "y": ws_sorted[0][1],
        "x0": ws_sorted[0][0],
        "x1": ws_sorted[-1][2],
        "text": txt,
        "words": ws_sorted,
    }


# =========================
# SECTION HEADER DETECTION
# =========================
def page_has_section_headers(text: str, section_name: str) -> bool:
    """
    Does this page contain the given section name AND the standard subheaders?
    (Key takeaway, Pros, Cons)
    """
    if not re.search(re.escape(section_name), text, re.I):
        return False
    return all(re.search(h, text, re.I) for h in SUBSECTION_HEADERS)


def find_section_header_y(page, section_name: str):
    """
    Locate the vertical position (y) of the header row for a given section on this page.
    Uses section name + any of the standard subheaders.
    """
    words = page.get_text("words") or []
    lines = group_words_into_lines(words)

    header_labels = [section_name] + SUBSECTION_HEADERS
    hdr_lines = [
        ln for ln in lines
        if any(lbl.lower() in ln["text"].lower() for lbl in header_labels)
    ]

    if hdr_lines:
        return min(hdr_lines, key=lambda x: x["y"])["y"]
    else:
        # Fallback: small margin from top
        _, h = page_size(page)
        return 0.10 * h


# =========================
# BULLET EXTRACTION
# =========================
def reconstruct_bullets(words, page_height):
    if not words:
        return []

    words_sorted = sorted(words, key=lambda w: (round(w["y0"], 1), w["x0"]))
    bullets, current = [], []
    started = False

    def flush():
        nonlocal current
        if current:
            bullets.append(" ".join(current).strip())
        current = []

    for w in words_sorted:
        t = normalize(w["text"])
        y = w["y0"]

        if not t:
            continue

        if is_header_footer(y, page_height, t):
            continue

        # Bullet start?
        if any(t.startswith(g) or t == g for g in BULLET_GLYPHS):
            started = True
            # Strip the bullet
            for g in BULLET_GLYPHS:
                if t.startswith(g):
                    t = t[len(g):].strip()
                elif t == g:
                    t = ""
            flush()
            if t:
                current.append(t)
        else:
            if not started:
                continue
            current.append(t)

    flush()
    return bullets


def get_prelude_text(words, page_height):
    """
    Join lines in this column that appear BEFORE the first bullet glyph,
    skipping dynamic headers/footers. Used mainly for continuation-page
    spillover lines that belong to the last bullet.
    """
    if not words:
        return ""

    lines = {}
    for w in words:
        ykey = round(w["y0"], 1)
        lines.setdefault(ykey, []).append(w)

    tokens = []
    for y in sorted(lines.keys()):
        line = sorted(lines[y], key=lambda k: k["x0"])
        text = " ".join(normalize(x["text"]) for x in line if x["text"])

        if is_header_footer(y, page_height, text):
            continue

        if any(g in text for g in BULLET_GLYPHS):
            break

        tokens.append(text)

    return " ".join(tokens).strip()


# =========================
# HEADER-LIKE TEXT DETECTION (for new sections such as ESG)
# =========================
def is_title_like_header(text: str) -> bool:
    """
    Heuristic to detect a section header such as 'ESG', 'Performance', etc.
    We avoid sentences and prefer short title-like lines.
    """
    t = normalize(text)
    if not t:
        return False

    # skip obvious subsection labels
    if any(sub.lower() in t.lower() for sub in SUBSECTION_HEADERS):
        return False

    # too long or looks like sentence
    if len(t) > 80:
        return False
    if "." in t:
        return False

    # tokenise
    tokens = re.findall(r"[A-Za-z][A-Za-z&'.-]*", t)
    if not tokens:
        return False
    if len(tokens) > 8:
        return False

    # require majority capitalised / uppercase
    caps = sum(1 for tok in tokens if tok[0].isupper() or tok.isupper())
    if caps / len(tokens) < 0.6:
        return False

    return True


def page_has_new_section_header(page, current_section: str) -> bool:
    """
    Detect any new *unknown* section header (e.g., 'ESG') near the top of the page.

    This is used ONLY after the 4th major section has started,
    to stop extraction when a new section begins.
    """
    w, h = page_size(page)
    words = page.get_text("words") or []
    lines = group_words_into_lines(words)

    for ln in lines:
        y = ln["y"]
        # scan band just below generic header zone, up to ~30% of page
        if y < 0.08 * h or y > 0.30 * h:
            continue

        text = ln["text"].strip()
        if not text:
            continue

        # ignore generic headers/footers if they somehow slip through
        if is_header_footer(y, h, text):
            continue

        lower = text.lower()

        # skip any line still referring to the current section name
        if current_section.lower() in lower:
            continue

        # skip if line includes any of the four known sections (we only care about unknown new sections)
        if any(sec.lower() in lower for sec in SECTIONS):
            continue

        if is_title_like_header(text):
            if PRINT_DEBUG:
                print(f"      New section header candidate on page: '{text}' at y={y:.1f}")
            return True

    return False


# =========================
# GRAPHICS-BASED DETECTION FOR PROS/CONS BAND
# =========================
def find_pros_cons_band_y(page, header_y: float):
    """
    Try to detect the coloured Pros/Cons header band using drawing objects.

    We look for wide rectangles a bit below the Key Takeaway area,
    which likely correspond to the blue Pros/Cons header bar.
    """
    w, h = page_size(page)
    try:
        draws = page.get_drawings()
    except Exception:
        return None

    cand_y = None
    for d in draws:
        for it in d.get("items", []):
            if it[0] == "re":  # rectangle
                r = it[1]
                # wide, moderate height, below Key Takeaway, upper half of page
                if r.width >= 0.5 * w and 8 <= r.height <= 80 and header_y + 5 < r.y0 < 0.60 * h:
                    if cand_y is None or r.y0 < cand_y:
                        cand_y = r.y0

    return cand_y


# =========================
# KEY TAKEAWAY INLINE (MULTI-LINE) EXTRACTION
# =========================
def extract_key_takeaway_from_page(page, section_name: str, header_y: float) -> str:
    """
    Multi-line Key Takeaway extraction (STRICT VERSION + WHITESPACE NORMALIZATION)

    STOP when:
        1) A line contains EXACT word 'Pros' or 'Cons' (after whitespace normalization)
        2) The Pros/Cons colored band is detected
        3) A bullet glyph appears (fallback)
    """
    w, h = page_size(page)
    y_bottom = bottom_y(page)
    words_raw = page.get_text("words") or []
    lines = group_words_into_lines(words_raw)

    # Detect Pros/Cons band (fallback)
    band_y = find_pros_cons_band_y(page, header_y)

    key_index = None
    initial_text_after_label = ""

    # ------------------------------------------------------
    # 1) FIND THE "Key takeaway:" LINE
    # ------------------------------------------------------
    for idx, ln in enumerate(lines):
        y = ln["y"]
        text = ln["text"]

        if y < header_y or y > y_bottom:
            continue
        if is_header_footer(y, h, text):
            continue

        if "key takeaway" not in text.lower():
            continue

        # Extract text after "Key takeaway:"
        m = re.search(r"key\s*takeaway\s*:?\s*(.*)", text, flags=re.I)
        if not m:
            continue

        remaining = m.group(1)

        # If Pros/Cons appear on the same line, cut before them
        m2 = re.search(r"\b(Pros|Cons)\b", remaining)
        if m2:
            remaining = remaining[:m2.start()]

        # If bullet appears, cut before it
        for g in BULLET_GLYPHS:
            if g in remaining:
                remaining = remaining[:remaining.find(g)]
                break

        initial_text_after_label = remaining.strip()
        key_index = idx
        break

    if key_index is None:
        return ""

    # ------------------------------------------------------
    # 2) COLLECT FOLLOWING LINES UNTIL STOP CONDITION
    # ------------------------------------------------------
    collected = []
    if initial_text_after_label:
        collected.append(initial_text_after_label)

    for ln in lines[key_index + 1:]:
        y = ln["y"]
        text = ln["text"]

        if y > y_bottom:
            break
        if is_header_footer(y, h, text):
            continue

        # NORMALIZE UNICODE WHITESPACE
        clean = normalize(text)     # converts NBSP, EM SPACE, etc. → " "
        tokens = clean.split()      # STRICT tokens

        # STRICT STOP: if line contains EXACT Pros or Cons
        if "Pros" in tokens or "Cons" in tokens:
            break

        # STOP at colored band (if detected)
        if band_y is not None and y >= band_y - 1:
            break

        # STOP at first bullet (Pros/Cons list)
        if any(g in text for g in BULLET_GLYPHS):
            break

        # Otherwise add the line
        if text.strip():
            collected.append(clean.strip())

    return " ".join(collected).strip()

# =========================
# PER PAGE PROS / CONS EXTRACTION
# =========================
def extract_from_page(page, continued=False, start_y=None):
    """
    Extract pros/cons bullets from a page:
      - if not continued, start just below the given header y (start_y)
      - if continued, start from a small top margin (carry-on page)
    Uses dynamic header/footer stripping and splits by mid-x.
    """
    w, h = page_size(page)
    y_bottom = bottom_y(page)

    words_raw = page.get_text("words") or []
    if not words_raw:
        return [], []

    if not continued:
        if start_y is None:
            start_y = 0.10 * h
        y_top = max(start_y + 2, 0.10 * h)
    else:
        y_top = 0.01 * h

    mid_x = w / 2.0
    left_words, right_words = [], []

    for wd in words_raw:
        y = wd[1]
        t = wd[4]

        if y < y_top or y > y_bottom:
            continue
        if is_header_footer(y, h, t):
            continue

        entry = {
            "x0": wd[0], "y0": wd[1], "x1": wd[2], "y1": wd[3],
            "text": t,
        }

        if (wd[0] + wd[2]) / 2.0 < mid_x:
            left_words.append(entry)
        else:
            right_words.append(entry)

    pros_words = left_words if PROS_ON_LEFT else right_words
    cons_words = right_words if PROS_ON_LEFT else left_words

    pros = reconstruct_bullets(pros_words, h)
    cons = reconstruct_bullets(cons_words, h)

    return pros, cons


# =========================
# SECTION-LEVEL EXTRACTION
# =========================
def extract_section(doc, section_name: str, start_page: int, end_page: int, is_last_section: bool = False):
    """
    Extract Key Takeaway, Pros, Cons for a given section
    between pages [start_page, end_page) of the document.

    For the last section (Business Management), we also dynamically
    stop when a new unknown section header (e.g. ESG) is encountered
    on any later page after we started collecting.
    """
    if PRINT_DEBUG:
        print(f"\n  Section '{section_name}' pages {start_page+1}–{end_page}")

    key_takeaway_parts = []
    all_pros, all_cons = [], []
    collecting = False
    header_y = None

    for page_index in range(start_page, end_page):
        page = doc[page_index]
        w, h = page_size(page)

        # For the LAST section, if we already started collecting,
        # stop when a new section header is detected on this page.
        if collecting and is_last_section and page_index > start_page:
            if page_has_new_section_header(page, section_name):
                if PRINT_DEBUG:
                    print(f"    New section header detected on page {page_index+1}; stopping '{section_name}' here.")
                break

        if not collecting:
            # First page for this section: find header y and grab key takeaway + bullets
            header_y = find_section_header_y(page, section_name)
            if PRINT_DEBUG:
                print(f"    Page {page_index+1}: header_y={header_y:.1f}")

            kt = extract_key_takeaway_from_page(page, section_name, header_y)
            if kt:
                key_takeaway_parts.append(kt)

            pros, cons = extract_from_page(page, continued=False, start_y=header_y)
            all_pros.extend(pros)
            all_cons.extend(cons)

            collecting = True
            continue

        # Continuation pages for this section
        if PRINT_DEBUG:
            print(f"    Page {page_index+1}: continuation")

        y_bottom = bottom_y(page)
        words_raw = page.get_text("words") or []

        mid_x = w / 2.0
        left_c, right_c = [], []

        for wd in words_raw:
            y = wd[1]
            t = wd[4]

            if y > y_bottom or is_header_footer(y, h, t):
                continue

            entry = {
                "x0": wd[0], "y0": wd[1],
                "x1": wd[2], "y1": wd[3],
                "text": t
            }

            if (wd[0] + wd[2]) / 2.0 < mid_x:
                left_c.append(entry)
            else:
                right_c.append(entry)

        # Spillover preludes (continuation lines before bullets)
        pros_pre = get_prelude_text(left_c, h)
        cons_pre = get_prelude_text(right_c, h)

        if pros_pre and all_pros:
            all_pros[-1] = (all_pros[-1] + " " + pros_pre).strip()
        if cons_pre and all_cons:
            all_cons[-1] = (all_cons[-1] + " " + cons_pre).strip()

        # New bullets
        pros, cons = extract_from_page(page, continued=True)
        all_pros.extend(pros)
        all_cons.extend(cons)

    key_takeaway = " ".join(key_takeaway_parts).strip()

    if PRINT_DEBUG:
        print(f"  → {section_name}: KT_len={len(key_takeaway)}, "
              f"Pros={len(all_pros)}, Cons={len(all_cons)}")

    return {
        "key_takeaway": key_takeaway,
        "pros": all_pros,
        "cons": all_cons,
    }


# =========================
# PROCESS ONE PDF
# =========================
def process_pdf(pdf_path, output_dir="output"):
    print(f"\nProcessing: {pdf_path}")
    os.makedirs(output_dir, exist_ok=True)

    with fitz.open(pdf_path) as doc:
        # --- Dynamic detection of repeated header/footer lines ---
        global FORBIDDEN_DYNAMIC
        FORBIDDEN_DYNAMIC = detect_repeating_header_footer_lines(doc)

        if PRINT_DEBUG:
            print("Detected dynamic header/footer lines:")
            for l in FORBIDDEN_DYNAMIC:
                print("  •", l)

        # 1) Find starting page index for each section (first occurrence)
        section_start_pages = {}
        for i, page in enumerate(doc):
            text = page.get_text("text") or ""
            for sec in SECTIONS:
                if sec in section_start_pages:
                    continue
                if page_has_section_headers(text, sec):
                    section_start_pages[sec] = i
                    if PRINT_DEBUG:
                        print(f"Found section '{sec}' starting on page {i+1}")

        if not section_start_pages:
            print("  ⚠ No matching sections found (no headers detected).")
            return

        sections_in_doc = [s for s in SECTIONS if s in section_start_pages]
        sections_data = {}

        # 2) Determine end page for each section:
        #    - until the next known section, OR
        #    - for the last one, until EOF (and then we dynamically stop inside extract_section).
        for idx, sec in enumerate(sections_in_doc):
            start_page = section_start_pages[sec]
            if idx + 1 < len(sections_in_doc):
                next_sec = sections_in_doc[idx + 1]
                end_page = section_start_pages[next_sec]
                is_last = False
            else:
                end_page = len(doc)
                is_last = True  # last of the four major sections

            sections_data[sec] = extract_section(doc, sec, start_page, end_page, is_last_section=is_last)

    # --- SAVE OUTPUT (single CSV, structured for RAG) ---
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    out_file = os.path.join(output_dir, base + "_sections_structured.csv")

    with open(out_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Section", "Subsection", "Content"])

        for sec in SECTIONS:
            data = sections_data.get(sec)
            if not data:
                continue

            # Key Takeaway
            kt = data.get("key_takeaway", "").strip()
            if kt:
                writer.writerow([sec, "Key Takeaway", kt])

            # Pros
            for p in data.get("pros", []):
                if p:
                    writer.writerow([sec, "Pros", p])

            # Cons
            for c in data.get("cons", []):
                if c:
                    writer.writerow([sec, "Cons", c])

    print(f"  ✓ Saved → {out_file}")


# =========================
# PROCESS ALL PDFs
# =========================
def main():
    # Adjust these paths as per your setup
    in_dir = "D:/Mercer/Data Extraction/21-11/input"
    out_dir = "D:/Mercer/Data Extraction/21-11/output"

    pdfs = [
        os.path.join(in_dir, f)
        for f in os.listdir(in_dir)
        if f.lower().endswith(".pdf")
    ]

    if not pdfs:
        print("No PDFs found.")
        return

    for p in pdfs:
        try:
            process_pdf(p, out_dir)
        except Exception as e:
            print(f"✖ Error processing {p}: {e}")


if __name__ == "__main__":
    main()
