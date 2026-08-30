"""Minimal deterministic PDF writer, used to build slide-deck test fixtures.

The project has no PDF *generation* dependency (and adding one just for tests
would be the wrong trade), so this module emits a tiny hand-written PDF with
base-14 Helvetica text. That is enough for pdfplumber/pdfminer to report the
text, its bounding box and its font size — the exact signals the ingestion
pipeline depends on — so the fixtures exercise the real extractor rather than
a mock.

    build_pdf([[line(...), line(...)], [line(...)]])  -> bytes

Coordinates are given in points from the TOP-LEFT of the page, which is how
slides are usually described and how pdfplumber reports them back.
"""
from __future__ import annotations

PAGE_WIDTH = 720.0
PAGE_HEIGHT = 540.0


def line(text: str, y: float, size: float = 14.0, x: float = 48.0, bold: bool = False) -> dict:
    """One text line, positioned y points below the top of the page."""
    return {"text": text, "y": y, "size": size, "x": x, "bold": bold}


def _escape(text: str) -> bytes:
    out = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
    return out.encode("cp1252", errors="replace")


def _content_stream(items: list[dict], height: float) -> bytes:
    parts = [b"BT"]
    for it in items:
        font = b"/F2" if it.get("bold") else b"/F1"
        baseline = height - it["y"] - it["size"]
        parts.append(
            b"%s %s Tf 1 0 0 1 %s %s Tm (%s) Tj"
            % (font, f"{it['size']:.2f}".encode(), f"{it['x']:.2f}".encode(),
               f"{baseline:.2f}".encode(), _escape(it["text"]))
        )
    parts.append(b"ET")
    return b"\n".join(parts)


def build_pdf(pages: list[list[dict]], width: float = PAGE_WIDTH,
              height: float = PAGE_HEIGHT) -> bytes:
    """Assemble the pages into a valid single-file PDF."""
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)  # 1-based object number

    n_pages = len(pages)
    catalog_num = 1
    pages_num = 2
    objects.append(b"")  # placeholder for catalog
    objects.append(b"")  # placeholder for /Pages
    font_regular = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
                       b"/Encoding /WinAnsiEncoding >>")
    font_bold = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold "
                    b"/Encoding /WinAnsiEncoding >>")

    page_nums: list[int] = []
    for items in pages:
        stream = _content_stream(items, height)
        content_num = add(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
        page_num = add(
            b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 %s %s] "
            b"/Resources << /Font << /F1 %d 0 R /F2 %d 0 R >> >> /Contents %d 0 R >>"
            % (pages_num, f"{width:.2f}".encode(), f"{height:.2f}".encode(),
               font_regular, font_bold, content_num)
        )
        page_nums.append(page_num)

    kids = b" ".join(b"%d 0 R" % n for n in page_nums)
    objects[pages_num - 1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, n_pages)
    objects[catalog_num - 1] = b"<< /Type /Catalog /Pages %d 0 R >>" % pages_num

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref_at = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += b"%010d 00000 n \n" % off
    out += (b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objects) + 1, catalog_num, xref_at))
    return bytes(out)


def spaced_line(text: str, y: float, size: float = 26.0, x: float = 48.0,
                bold: bool = True, tracking: float = 6.0) -> list[dict]:
    """A title typeset with character spacing, one glyph at a time.

    Slide titles are often set this way, and extractors then report a run of
    single-character "words". Positions here use the real base-14 metrics plus
    a constant tracking value, so the word gaps stay genuinely wider than the
    letter gaps — the same signal the ingestion repair relies on in real files.
    """
    from pdfminer.fontmetrics import FONT_METRICS

    _, widths = FONT_METRICS["Helvetica-Bold" if bold else "Helvetica"]
    out = []
    cursor = x
    for ch in text:
        advance = widths.get(ch, 500) * size / 1000 + tracking
        if ch != " ":
            out.append(line(ch, y, size, x=cursor, bold=bold))
        cursor += advance
    return out
