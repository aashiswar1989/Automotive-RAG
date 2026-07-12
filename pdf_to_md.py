"""
Convert the braking requirements PDF into structure-aware Markdown.

Why not just dump raw text?
The source PDF has no embedded outline/bookmarks and MarkItDown's default
PDF backend returns flat paragraphs with no '#' headers. But the document
is extremely regular:

    1. Scope                              -> H1 section
    3.1 Legacy Hydraulic Architecture      -> H2 subsection
    REQ-SYS-001 - Service Brake Provision  -> H3 requirement item

Exploiting that regularity to inject real markdown headers gives every
downstream chunking strategy (especially MarkdownHeaderTextSplitter) a
much stronger signal than blind character splitting would.
"""

import re
import unicodedata
from pathlib import Path

from markitdown import MarkItDown

SECTION_RE = re.compile(r"^(\d{1,2})\.\s+(.+)$")
SUBSECTION_RE = re.compile(r"^(\d{1,2}\.\d{1,2})\s+(.+)$")
REQ_RE = re.compile(r"^(REQ-[A-Z]+-\d{3})\s*[–-]\s*(.+)$")


def clean_text(raw: str) -> str:
    """Strip PDF page-break artifacts and normalize ligature glyphs.

    Two distinct ligature problems show up in this PDF's font encoding:
    - fi/fl/ffi/ffl etc. extract as precomposed Unicode ligature codepoints
      (e.g. 'ﬁ') -> fixed by NFKC normalization.
    - 'ft' has no Unicode ligature codepoint at all in this font, so it
      extracts as a bare NUL byte (e.g. 'a\\x00er' for 'after') -> must be
      fixed with an explicit replacement, NFKC won't touch it.
    """
    text = raw.replace("\f", "\n")
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "ft")
    return text


_SENTENCE_END_RE = re.compile(r"[.!?](\[\w+:\d+\])*$")


def reflow_paragraphs(text: str) -> str:
    """Merge PDF hard line-wraps back into paragraphs / list items.

    MarkItDown's blank-line spacing in this PDF is NOT a reliable paragraph
    boundary: multi-line prose paragraphs and single-line bulleted list
    items both get blank-line-separated the same way. Instead we merge
    lines using a sentence-terminal heuristic: keep appending lines to the
    current unit until one ends in '.', '!', or '?' (optionally followed
    by citation markers like '[web:7]'). Headers always start a new unit.
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    units, buf = [], []
    for line in lines:
        if line.startswith("#"):
            if buf:
                units.append(" ".join(buf))
                buf = []
            units.append(line)
            continue
        buf.append(line)
        if _SENTENCE_END_RE.search(line):
            units.append(" ".join(buf))
            buf = []
    if buf:
        units.append(" ".join(buf))

    return "\n\n".join(units)


def to_markdown(raw: str) -> str:
    cleaned = clean_text(raw)
    lines = [ln.strip() for ln in cleaned.split("\n")]

    out = []
    for line in lines:
        if not line:
            out.append("")
            continue

        m = REQ_RE.match(line)
        if m:
            req_id, title = m.groups()
            out.append(f"### {req_id} – {title}")
            continue

        m = SUBSECTION_RE.match(line)
        if m:
            num, title = m.groups()
            out.append(f"## {num} {title}")
            continue

        m = SECTION_RE.match(line)
        if m:
            num, title = m.groups()
            out.append(f"# {num}. {title}")
            continue

        out.append(line)

    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = reflow_paragraphs(text)
    return text.strip() + "\n"


def convert(pdf_path: Path, out_path: Path) -> Path:
    md_engine = MarkItDown()
    result = md_engine.convert(str(pdf_path))
    markdown = to_markdown(result.text_content)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    src = Path("data/raw/braking_system_requirements_merged.pdf")
    dst = Path("data/processed/braking_system_requirements.md")
    convert(src, dst)
    print(f"wrote {dst} ({dst.stat().st_size} bytes)")
