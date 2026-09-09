"""HaqBot — offline ingestion of UAE labour-law PDFs into cited, chunked text.

Pipeline (100% on-device, no network):

    PDF file
      -> PyMuPDF page-text extraction
      -> whitespace / hyphenation cleanup, running header-footer stripping
      -> structural parse: CHAPTER / PART, ``Article (N)`` + title, numbered and
         lettered clauses
      -> RecursiveCharacterTextSplitter (chunk_size=450, overlap=50) per article
      -> :class:`Chunk` objects carrying full citation metadata
      -> JSONL persisted at ``data/processed/chunks.jsonl``

Every failure raises :class:`IngestionError` with a plain, user-safe message;
no raw traceback ever reaches the UI.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import pymupdf

from src import config

if TYPE_CHECKING:  # heavy (pulls langchain_core); imported lazily at call time
    from langchain_text_splitters import RecursiveCharacterTextSplitter

__all__ = [
    "IngestionError",
    "PageText",
    "ClauseSpan",
    "ArticleSegment",
    "Chunk",
    "clean_text",
    "extract_pages",
    "resolve_source",
    "parse_articles",
    "build_splitter",
    "chunk_segment",
    "format_citation",
    "ingest_pdf",
    "ingest_directory",
    "write_chunks_jsonl",
    "read_chunks_jsonl",
]


class IngestionError(Exception):
    """Any problem opening, parsing, or chunking a source document."""


# =========================================================================== #
# Regexes
# =========================================================================== #
_CHAPTER_RE = re.compile(
    r"^[ \t]*(CHAPTER|PART|TITLE|SECTION|BOOK)[ \t]+([A-Z0-9IVXLM]+)\b"
    r"[ \t]*[:.\-–—]?[ \t]*(.*)$",
    re.MULTILINE | re.IGNORECASE,
)
_ARTICLE_RE = re.compile(
    r"^[ \t]*Article[ \t]*\(?[ \t]*(\d+)[ \t]*(bis|ter)?[ \t]*\)?"
    r"[ \t]*[:.\-–—]?[ \t]*(.*)$",
    re.MULTILINE | re.IGNORECASE,
)
# A clause opener: "1." / "(2)" / "3)" / "a." / "(b)" at the start of a line.
_CLAUSE_RE = re.compile(r"^[ \t]*\(?([0-9]{1,2}|[a-z])\)?[.)][ \t]+(?=\S)", re.MULTILINE)

_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")
_MANY_NEWLINES_RE = re.compile(r"\n{3,}")
_PAGE_NUMBER_LINE_RE = re.compile(
    r"^[ \t]*(?:page[ \t]+)?\d{1,4}[ \t]*$", re.IGNORECASE | re.MULTILINE
)


# =========================================================================== #
# Data structures
# =========================================================================== #
@dataclass(frozen=True)
class PageText:
    """Raw text of a single PDF page (1-based number)."""

    page_number: int
    text: str


@dataclass(frozen=True)
class ClauseSpan:
    """A clause's identifier and its character span within an article body."""

    number: str
    start: int
    end: int


@dataclass
class ArticleSegment:
    """One article (or an unstructured document body when no articles exist)."""

    source_id: str
    source_title: str
    article_number: str | None
    article_title: str | None
    chapter: str | None
    section: str | None
    body: str
    clauses: list[ClauseSpan] = field(default_factory=list)
    page_number: int | None = None


@dataclass
class Chunk:
    """A retrieval unit: chunk text plus citation metadata."""

    text: str
    metadata: dict[str, Any]

    @property
    def citation(self) -> str:
        return self.metadata.get("citation", "")

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(text=data["text"], metadata=dict(data["metadata"]))

    def to_document(self):
        """Return a LangChain ``Document`` (imported lazily to stay light)."""
        from langchain_core.documents import Document

        return Document(page_content=self.text, metadata=dict(self.metadata))


# =========================================================================== #
# Cleaning
# =========================================================================== #
def clean_text(raw: str) -> str:
    """Normalise extracted PDF text without destroying line structure."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("­", "")                    # soft hyphen
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)           # de-hyphenate at breaks
    text = _PAGE_NUMBER_LINE_RE.sub("", text)            # drop "12" / "Page 3"
    text = _MULTISPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MANY_NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def _strip_running_headers(pages: list[PageText]) -> list[PageText]:
    """Remove first/last lines that repeat (verbatim) across most pages."""
    if len(pages) < 3:
        return pages
    first_lines: dict[str, int] = {}
    last_lines: dict[str, int] = {}
    for pg in pages:
        lines = [ln.strip() for ln in pg.text.splitlines() if ln.strip()]
        if not lines:
            continue
        first_lines[lines[0]] = first_lines.get(lines[0], 0) + 1
        last_lines[lines[-1]] = last_lines.get(lines[-1], 0) + 1
    threshold = max(2, int(len(pages) * 0.6))
    drop_first = {k for k, v in first_lines.items() if v >= threshold}
    drop_last = {k for k, v in last_lines.items() if v >= threshold}
    if not drop_first and not drop_last:
        return pages
    cleaned: list[PageText] = []
    for pg in pages:
        lines = pg.text.splitlines()
        while lines and lines[0].strip() in drop_first:
            lines.pop(0)
        while lines and lines[-1].strip() in drop_last:
            lines.pop()
        cleaned.append(PageText(pg.page_number, "\n".join(lines)))
    return cleaned


# =========================================================================== #
# Extraction
# =========================================================================== #
def extract_pages(pdf_path: str | Path) -> list[PageText]:
    """Return per-page text for a PDF. Raises :class:`IngestionError`."""
    path = Path(pdf_path)
    if not path.is_file():
        raise IngestionError(f"PDF not found: {path.name}")
    try:
        doc = pymupdf.open(path)
    except Exception as exc:  # noqa: BLE001 - surface a clean message only
        raise IngestionError(f"Could not open PDF '{path.name}': {exc}") from exc
    try:
        pages = [
            PageText(page_number=i + 1, text=page.get_text("text"))
            for i, page in enumerate(doc)
        ]
    finally:
        doc.close()
    if not any(p.text.strip() for p in pages):
        raise IngestionError(
            f"'{path.name}' has no extractable text (is it a scanned image?)."
        )
    return pages


def resolve_source(pdf_path: str | Path) -> tuple[str, str]:
    """Map a PDF path to ``(source_id, canonical_title)``."""
    stem = Path(pdf_path).stem
    key = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    for needle, title in config.KNOWN_LEGAL_SOURCES:
        if needle in key:
            return stem, title
    fallback = re.sub(r"[-_]+", " ", stem).strip().title() or "Untitled Document"
    return stem, fallback


# =========================================================================== #
# Structural parse
# =========================================================================== #
def _find_clause_spans(body: str) -> list[ClauseSpan]:
    marks = [(m.start(), m.group(1)) for m in _CLAUSE_RE.finditer(body)]
    spans: list[ClauseSpan] = []
    for idx, (start, number) in enumerate(marks):
        end = marks[idx + 1][0] if idx + 1 < len(marks) else len(body)
        spans.append(ClauseSpan(number=number, start=start, end=end))
    return spans


def _clauses_in_span(spans: Iterable[ClauseSpan], start: int, end: int) -> list[str]:
    hit: list[str] = []
    for span in spans:
        if span.start < end and span.end > start and span.number not in hit:
            hit.append(span.number)
    return hit


def parse_articles(
    pages: list[PageText], source_id: str, source_title: str
) -> list[ArticleSegment]:
    """Split a document into article segments with chapter / clause metadata."""
    pages = _strip_running_headers(pages)

    parts: list[str] = []
    page_marks: list[tuple[int, int]] = []   # (char_offset, page_number)
    cursor = 0
    for pg in pages:
        cleaned = clean_text(pg.text)
        page_marks.append((cursor, pg.page_number))
        parts.append(cleaned)
        cursor += len(cleaned) + 2           # "\n\n" join
    full = "\n\n".join(parts)

    def page_for(offset: int) -> int | None:
        found = pages[0].page_number if pages else None
        for off, num in page_marks:
            if off <= offset:
                found = num
            else:
                break
        return found

    chapters = [
        (
            m.start(),
            f"{m.group(1).title()} {m.group(2).upper()}".strip(),
            (m.group(3) or "").strip(),
        )
        for m in _CHAPTER_RE.finditer(full)
    ]

    def chapter_for(offset: int) -> tuple[str, str] | None:
        current: tuple[str, str] | None = None
        for off, label, title in chapters:
            if off <= offset:
                current = (label, title)
            else:
                break
        return current

    matches = list(_ARTICLE_RE.finditer(full))
    if not matches:
        body = full.strip()
        if len(body) < config.MIN_CHUNK_CHARS:
            return []
        return [
            ArticleSegment(
                source_id=source_id,
                source_title=source_title,
                article_number=None,
                article_title=None,
                chapter=None,
                section=None,
                body=body,
                clauses=_find_clause_spans(body),
                page_number=page_for(0),
            )
        ]

    segments: list[ArticleSegment] = []
    for i, m in enumerate(matches):
        article_number = m.group(1)
        article_title = (m.group(3) or "").strip() or None
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(full)
        body = full[body_start:body_end].strip()

        # Title often sits on the line *after* the "Article (N)" header.
        if article_title is None and body:
            first_line, _sep, rest = body.partition("\n")
            candidate = first_line.strip()
            if (
                candidate
                and len(candidate) <= 80
                and not candidate[0].isdigit()
                and not _CLAUSE_RE.match(candidate)
            ):
                article_title = candidate
                body = rest.strip()

        chapter = chapter_for(m.start())
        segments.append(
            ArticleSegment(
                source_id=source_id,
                source_title=source_title,
                article_number=article_number,
                article_title=article_title,
                chapter=chapter[0] if chapter else None,
                section=(chapter[1] or None) if chapter else None,
                body=body,
                clauses=_find_clause_spans(body),
                page_number=page_for(m.start()),
            )
        )
    return segments


# =========================================================================== #
# Citations
# =========================================================================== #
def format_citation(
    source_title: str,
    article_number: str | None = None,
    clause_numbers: list[str] | None = None,
) -> str:
    """Build a bracketed citation, e.g. ``[Article 43, Clause 2]``."""
    if not article_number:
        return f"[{source_title}]"
    clause_numbers = list(clause_numbers or [])
    if not clause_numbers:
        return f"[Article {article_number}]"
    if len(clause_numbers) == 1:
        return f"[Article {article_number}, Clause {clause_numbers[0]}]"
    if all(c.isdigit() for c in clause_numbers):
        nums = sorted(int(c) for c in clause_numbers)
        if nums == list(range(nums[0], nums[-1] + 1)):
            return f"[Article {article_number}, Clauses {nums[0]}–{nums[-1]}]"
    return f"[Article {article_number}, Clauses {', '.join(clause_numbers)}]"


# =========================================================================== #
# Chunking
# =========================================================================== #
def build_splitter(
    chunk_size: int | None = None, chunk_overlap: int | None = None
) -> "RecursiveCharacterTextSplitter":
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or config.CHUNK_SIZE,
        chunk_overlap=(
            config.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
        ),
        separators=list(config.CHUNK_SEPARATORS),
        keep_separator=True,
        add_start_index=True,
    )


def chunk_segment(
    segment: ArticleSegment, splitter: "RecursiveCharacterTextSplitter"
) -> list[Chunk]:
    """Split one article body into overlapping chunks with metadata."""
    body = segment.body.strip()
    if not body:
        return []
    docs = splitter.create_documents([body])
    chunks: list[Chunk] = []
    for chunk_index, doc in enumerate(docs):
        piece = doc.page_content.strip()
        if not piece:
            continue
        if len(piece) < config.MIN_CHUNK_CHARS and len(docs) > 1:
            continue
        start = int(doc.metadata.get("start_index", body.find(piece)))
        if start < 0:
            start = 0
        end = start + len(doc.page_content)
        clause_numbers = _clauses_in_span(segment.clauses, start, end)
        metadata: dict[str, Any] = {
            "chunk_id": (
                f"{segment.source_id}::art-"
                f"{segment.article_number or '0'}::{chunk_index}"
            ),
            "source_id": segment.source_id,
            "source_title": segment.source_title,
            "doc_title": segment.source_title,
            "article_number": segment.article_number,
            "article_title": segment.article_title,
            "clause_numbers": clause_numbers,
            "chapter": segment.chapter,
            "section": segment.section,
            "page_number": segment.page_number,
            "chunk_index": chunk_index,
            "char_start": start,
            "char_end": end,
            "citation": format_citation(
                segment.source_title, segment.article_number, clause_numbers
            ),
            "language": config.INGEST_LANGUAGE,
        }
        chunks.append(Chunk(text=piece, metadata=metadata))
    return chunks


# =========================================================================== #
# Orchestration
# =========================================================================== #
def ingest_pdf(
    pdf_path: str | Path,
    *,
    splitter: "RecursiveCharacterTextSplitter | None" = None,
) -> list[Chunk]:
    """Full pipeline for a single PDF. Raises :class:`IngestionError`."""
    splitter = splitter or build_splitter()
    pages = extract_pages(pdf_path)
    source_id, source_title = resolve_source(pdf_path)
    segments = parse_articles(pages, source_id, source_title)
    chunks: list[Chunk] = []
    for segment in segments:
        chunks.extend(chunk_segment(segment, splitter))
    if not chunks:
        raise IngestionError(
            f"No usable text could be chunked from '{Path(pdf_path).name}'."
        )
    return chunks


def ingest_directory(
    raw_dir: str | Path | None = None,
    *,
    splitter: "RecursiveCharacterTextSplitter | None" = None,
) -> list[Chunk]:
    """Ingest every ``*.pdf`` in a directory (sorted). Raises if none found."""
    directory = Path(raw_dir or config.RAW_DIR)
    pdfs = sorted(directory.glob("*.pdf"))
    if not pdfs:
        raise IngestionError(f"No PDF files found in {directory}")
    splitter = splitter or build_splitter()
    all_chunks: list[Chunk] = []
    for pdf in pdfs:
        all_chunks.extend(ingest_pdf(pdf, splitter=splitter))
    return all_chunks


# =========================================================================== #
# Persistence
# =========================================================================== #
def write_chunks_jsonl(
    chunks: list[Chunk], path: str | Path | None = None
) -> Path:
    """Write chunks as one JSON object per line."""
    out_path = Path(path or config.CHUNKS_META_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    return out_path


def read_chunks_jsonl(path: str | Path | None = None) -> list[Chunk]:
    """Read chunks written by :func:`write_chunks_jsonl`."""
    in_path = Path(path or config.CHUNKS_META_PATH)
    if not in_path.is_file():
        raise IngestionError(f"Chunk file not found: {in_path}")
    chunks: list[Chunk] = []
    with in_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(Chunk.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError) as exc:
                raise IngestionError(
                    f"Corrupt chunk record at line {line_no} of {in_path.name}."
                ) from exc
    return chunks


# =========================================================================== #
# CLI: python -m src.ingestion  [raw_dir]  [out.jsonl]
# =========================================================================== #
def _main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI
    import argparse

    parser = argparse.ArgumentParser(description="Ingest UAE labour-law PDFs.")
    parser.add_argument("raw_dir", nargs="?", default=str(config.RAW_DIR))
    parser.add_argument("out", nargs="?", default=str(config.CHUNKS_META_PATH))
    args = parser.parse_args(argv)
    try:
        chunks = ingest_directory(args.raw_dir)
    except IngestionError as exc:
        print(f"ingestion failed: {exc}")
        return 1
    out_path = write_chunks_jsonl(chunks, args.out)
    by_source: dict[str, int] = {}
    for chunk in chunks:
        by_source[chunk.metadata["source_title"]] = (
            by_source.get(chunk.metadata["source_title"], 0) + 1
        )
    print(f"wrote {len(chunks)} chunks -> {out_path}")
    for title, count in by_source.items():
        print(f"  {count:5d}  {title}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
