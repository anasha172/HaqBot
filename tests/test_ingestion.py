"""Phase 3 — PDF ingestion, structural parsing, chunking & citation metadata.

No real PDFs ship with the repo, so each test synthesises a small PDF with
PyMuPDF (offline, no fonts fetched) whose Article / Clause structure is known.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

import pymupdf
import pytest

from src import config, ingestion
from src.ingestion import (
    Chunk,
    IngestionError,
    build_splitter,
    chunk_segment,
    clean_text,
    extract_pages,
    format_citation,
    ingest_directory,
    ingest_pdf,
    parse_articles,
    read_chunks_jsonl,
    resolve_source,
    write_chunks_jsonl,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
LEGAL_TEXT = """\
Federal Decree-Law No. 33 of 2021
On the Regulation of Labour Relations

CHAPTER 2 - Employment Contracts and Records

Article (1)
Definitions
In the application of the provisions of this Decree-Law, the following
words and expressions shall have the meanings assigned to each of them,
unless the context requires otherwise.

Article (9) - Probationary Period
1. The employer may place the worker on probation for a period not
exceeding six months from the date of commencement of the work.
2. Where the employer terminates the worker during the probationary
period, the employer shall notify the worker at least fourteen days
before the date set for termination.
3. Where the worker wishes to move to another employer during the
probationary period, the worker shall notify the original employer in
writing at least one month in advance.

CHAPTER 8 - End of Service Entitlements

Article (51)
End of Service Gratuity
1. A full-time foreign worker who has completed one year or more of
continuous service shall be entitled to an end of service gratuity
upon the termination of the service.
2. The end of service gratuity shall be calculated as follows:
a. A wage of twenty-one days for each year of the first five years.
b. A wage of thirty days for each year exceeding the first five years.
3. The gratuity shall not exceed the total wage of two years.
"""

WPS_TEXT = """\
Wage Protection System Regulations

Article (2)
Scope of Application
The provisions of this resolution shall apply to all establishments
registered with the Ministry that are subject to the Wage Protection
System through approved agents.

Article (5) - Payment Timeline
1. Employers shall pay wages within fifteen days from the due date.
2. An employer who delays payment beyond the period is considered late.
"""


def _make_pdf(path, text, *, lines_per_page=40):
    """Render plain text into a real multi-page PDF (short lines, no wrapping)."""
    doc = pymupdf.open()
    lines = text.split("\n")
    for offset in range(0, len(lines), lines_per_page):
        page = doc.new_page()
        y = 60.0
        for line in lines[offset:offset + lines_per_page]:
            page.insert_text((56, y), line, fontsize=10, fontname="helv")
            y += 16.0
    doc.save(path)
    doc.close()


@pytest.fixture
def decree_pdf(tmp_path):
    path = tmp_path / "federal-decree-law-33-2021.pdf"
    _make_pdf(path, LEGAL_TEXT)
    return path


@pytest.fixture
def wps_pdf(tmp_path):
    path = tmp_path / "wps-regulations.pdf"
    _make_pdf(path, WPS_TEXT)
    return path


# --------------------------------------------------------------------------- #
# clean_text
# --------------------------------------------------------------------------- #
class TestCleanText:
    def test_dehyphenates_across_line_breaks(self):
        assert "gratuity" in clean_text("grat-\nuity due")

    def test_collapses_runs_of_spaces_and_blank_lines(self):
        out = clean_text("a    b\n\n\n\nc")
        assert "a b" in out
        assert "\n\n\n" not in out

    def test_drops_bare_page_number_lines(self):
        out = clean_text("Article (3)\n\n12\n\nText body here")
        assert re.search(r"^\s*12\s*$", out, re.MULTILINE) is None

    def test_strips_soft_hyphen(self):
        assert clean_text("wag­e") == "wage"


# --------------------------------------------------------------------------- #
# extract_pages / resolve_source
# --------------------------------------------------------------------------- #
class TestExtract:
    def test_extract_pages_numbers_from_one(self, decree_pdf):
        pages = extract_pages(decree_pdf)
        assert pages and pages[0].page_number == 1
        assert any("Probationary Period" in p.text for p in pages)

    def test_missing_file_raises_clean_error(self, tmp_path):
        with pytest.raises(IngestionError):
            extract_pages(tmp_path / "nope.pdf")

    def test_corrupt_pdf_raises_clean_error(self, tmp_path):
        bad = tmp_path / "broken.pdf"
        bad.write_bytes(b"%PDF-1.4 not really a pdf")
        with pytest.raises(IngestionError):
            extract_pages(bad)

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("federal-decree-law-33-2021.pdf", "Federal Decree-Law No. 33 of 2021"),
            ("UAE_Labour_Law.pdf", "Federal Decree-Law No. 33 of 2021"),
            ("wps-regulations.pdf", "Wage Protection System (WPS) Regulations"),
            ("mohre-circular-7.pdf", "MOHRE Ministerial Directives"),
        ],
    )
    def test_resolve_known_sources(self, name, expected):
        source_id, title = resolve_source(name)
        assert title == expected
        assert source_id == name[:-4]

    def test_resolve_unknown_source_falls_back_to_stem(self):
        _sid, title = resolve_source("some_random-file.pdf")
        assert title == "Some Random File"


# --------------------------------------------------------------------------- #
# parse_articles
# --------------------------------------------------------------------------- #
class TestParseArticles:
    @pytest.fixture
    def segments(self, decree_pdf):
        pages = extract_pages(decree_pdf)
        return parse_articles(pages, *resolve_source(decree_pdf))

    def test_finds_all_article_numbers(self, segments):
        assert [s.article_number for s in segments] == ["1", "9", "51"]

    def test_captures_title_on_same_line(self, segments):
        art9 = next(s for s in segments if s.article_number == "9")
        assert art9.article_title == "Probationary Period"

    def test_captures_title_on_following_line(self, segments):
        art51 = next(s for s in segments if s.article_number == "51")
        assert art51.article_title == "End of Service Gratuity"
        assert not art51.body.startswith("End of Service Gratuity")

    def test_tracks_chapter_context(self, segments):
        art9 = next(s for s in segments if s.article_number == "9")
        art51 = next(s for s in segments if s.article_number == "51")
        assert art9.chapter == "Chapter 2"
        assert "Employment Contracts" in (art9.section or "")
        assert art51.chapter == "Chapter 8"

    def test_detects_numbered_and_lettered_clauses(self, segments):
        art51 = next(s for s in segments if s.article_number == "51")
        numbers = [c.number for c in art51.clauses]
        assert numbers == ["1", "2", "a", "b", "3"]

    def test_clause_spans_are_ordered_and_bounded(self, segments):
        art9 = next(s for s in segments if s.article_number == "9")
        prev_end = -1
        for clause in art9.clauses:
            assert 0 <= clause.start < clause.end <= len(art9.body)
            assert clause.start >= prev_end
            prev_end = clause.start

    def test_document_without_articles_becomes_one_segment(self, tmp_path):
        path = tmp_path / "notice.pdf"
        _make_pdf(path, "General Notice\n\n" + "This is guidance text. " * 20)
        segs = parse_articles(extract_pages(path), *resolve_source(path))
        assert len(segs) == 1
        assert segs[0].article_number is None

    def test_tiny_document_yields_no_segments(self, tmp_path):
        path = tmp_path / "stub.pdf"
        _make_pdf(path, "Notice.")
        assert parse_articles(extract_pages(path), *resolve_source(path)) == []


class TestRunningHeaders:
    def test_repeated_first_and_last_lines_removed(self):
        pages = [
            ingestion.PageText(
                n,
                "MINISTRY OF HUMAN RESOURCES\n"
                f"Article ({n}) Body text for page {n}.\n"
                "www.mohre.gov.ae",
            )
            for n in range(1, 6)
        ]
        stripped = ingestion._strip_running_headers(pages)
        joined = "\n".join(p.text for p in stripped)
        assert "MINISTRY OF HUMAN RESOURCES" not in joined
        assert "www.mohre.gov.ae" not in joined
        assert "Body text for page 3" in joined

    def test_no_op_when_headers_do_not_repeat(self):
        pages = [ingestion.PageText(n, f"Unique line {n}") for n in range(1, 5)]
        assert ingestion._strip_running_headers(pages) == pages

    def test_scanned_pdf_without_text_raises(self, tmp_path):
        path = tmp_path / "scan.pdf"
        doc = pymupdf.open()
        doc.new_page()
        doc.new_page()
        doc.save(path)
        doc.close()
        with pytest.raises(IngestionError):
            extract_pages(path)


# --------------------------------------------------------------------------- #
# format_citation
# --------------------------------------------------------------------------- #
class TestFormatCitation:
    def test_source_only(self):
        assert format_citation("WPS Regulations") == "[WPS Regulations]"

    def test_article_only(self):
        assert format_citation("X", "43") == "[Article 43]"

    def test_single_clause(self):
        assert format_citation("X", "43", ["2"]) == "[Article 43, Clause 2]"

    def test_contiguous_clause_range(self):
        assert format_citation("X", "9", ["1", "2", "3"]) == "[Article 9, Clauses 1–3]"

    def test_non_contiguous_clauses(self):
        assert format_citation("X", "9", ["1", "3"]) == "[Article 9, Clauses 1, 3]"

    def test_lettered_clauses_listed(self):
        assert format_citation("X", "51", ["a", "b"]) == "[Article 51, Clauses a, b]"


# --------------------------------------------------------------------------- #
# chunk_segment
# --------------------------------------------------------------------------- #
class TestChunkSegment:
    @pytest.fixture
    def chunks(self, decree_pdf):
        return ingest_pdf(decree_pdf)

    def test_every_chunk_has_full_metadata(self, chunks):
        required = {
            "chunk_id", "source_id", "source_title", "doc_title",
            "article_number", "article_title", "clause_numbers", "chapter",
            "section", "page_number", "chunk_index", "char_start", "char_end",
            "citation", "language",
        }
        for chunk in chunks:
            assert required <= set(chunk.metadata)
            assert chunk.metadata["char_start"] < chunk.metadata["char_end"]
            assert chunk.metadata["language"] == "en"
            assert chunk.text.strip() == chunk.text

    def test_chunk_ids_are_unique(self, chunks):
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunks_respect_size_budget(self, chunks):
        # RecursiveCharacterTextSplitter can slightly overshoot with separators.
        assert all(len(c.text) <= config.CHUNK_SIZE + 120 for c in chunks)
        assert len(chunks) >= 3

    def test_citations_are_well_formed(self, chunks):
        pattern = re.compile(
            r"^\[(?:Article \d+(?:, Clauses? [0-9a-z, –-]+)?|[^\]]+)\]$"
        )
        for chunk in chunks:
            assert pattern.match(chunk.metadata["citation"]), chunk.metadata

    def test_article_9_chunks_carry_article_9_clauses(self, chunks):
        art9 = [c for c in chunks if c.metadata["article_number"] == "9"]
        assert art9
        for chunk in art9:
            assert chunk.metadata["citation"].startswith("[Article 9")
            assert set(chunk.metadata["clause_numbers"]) <= {"1", "2", "3"}

    def test_gratuity_query_area_cites_article_51(self, chunks):
        hits = [c for c in chunks if "twenty-one days" in c.text]
        assert hits
        assert hits[0].metadata["article_number"] == "51"
        assert "Clause" in hits[0].metadata["citation"]

    def test_small_trailing_fragments_dropped(self):
        seg = ingestion.ArticleSegment(
            source_id="x", source_title="X", article_number="1",
            article_title=None, chapter=None, section=None,
            body="A" * 460 + "\n\ntiny",
        )
        pieces = chunk_segment(seg, build_splitter())
        assert all(len(p.text) >= config.MIN_CHUNK_CHARS for p in pieces)

    def test_empty_segment_body_yields_nothing(self):
        seg = ingestion.ArticleSegment(
            source_id="x", source_title="X", article_number="4",
            article_title=None, chapter=None, section=None, body="   ",
        )
        assert chunk_segment(seg, build_splitter()) == []

    def test_chunk_citation_property(self):
        chunk = Chunk(text="body", metadata={"citation": "[Article 7, Clause 1]"})
        assert chunk.citation == "[Article 7, Clause 1]"
        assert Chunk(text="body", metadata={}).citation == ""


# --------------------------------------------------------------------------- #
# Orchestration + persistence
# --------------------------------------------------------------------------- #
class TestOrchestration:
    def test_ingest_directory_covers_all_pdfs(self, decree_pdf, wps_pdf):
        chunks = ingest_directory(decree_pdf.parent)
        sources = {c.metadata["source_id"] for c in chunks}
        assert sources == {decree_pdf.stem, wps_pdf.stem}

    def test_ingest_directory_empty_raises(self, tmp_path):
        with pytest.raises(IngestionError):
            ingest_directory(tmp_path)

    def test_ingest_pdf_missing_raises(self, tmp_path):
        with pytest.raises(IngestionError):
            ingest_pdf(tmp_path / "ghost.pdf")

    def test_jsonl_roundtrip_is_lossless(self, decree_pdf, tmp_path):
        chunks = ingest_pdf(decree_pdf)
        out = write_chunks_jsonl(chunks, tmp_path / "chunks.jsonl")
        assert out.is_file()
        restored = read_chunks_jsonl(out)
        assert len(restored) == len(chunks)
        for a, b in zip(chunks, restored):
            assert a.text == b.text
            assert a.metadata == b.metadata

    def test_jsonl_is_utf8_one_object_per_line(self, decree_pdf, tmp_path):
        out = write_chunks_jsonl(ingest_pdf(decree_pdf), tmp_path / "c.jsonl")
        for line in out.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            assert set(record) == {"text", "metadata"}

    def test_read_missing_jsonl_raises(self, tmp_path):
        with pytest.raises(IngestionError):
            read_chunks_jsonl(tmp_path / "absent.jsonl")

    def test_blank_lines_in_jsonl_are_skipped(self, tmp_path):
        path = tmp_path / "sparse.jsonl"
        path.write_text(
            '{"text": "a", "metadata": {}}\n\n   \n'
            '{"text": "b", "metadata": {}}\n',
            encoding="utf-8",
        )
        assert [c.text for c in read_chunks_jsonl(path)] == ["a", "b"]

    def test_ingest_pdf_raises_when_nothing_chunks(self, decree_pdf, monkeypatch):
        monkeypatch.setattr(ingestion, "parse_articles", lambda *a, **k: [])
        with pytest.raises(IngestionError):
            ingest_pdf(decree_pdf)

    def test_corrupt_jsonl_raises_clean_error(self, tmp_path):
        bad = tmp_path / "bad.jsonl"
        bad.write_text('{"text": "ok"}\nnot json\n', encoding="utf-8")
        with pytest.raises(IngestionError):
            read_chunks_jsonl(bad)

    def test_chunk_to_document_shape(self, decree_pdf):
        chunk = ingest_pdf(decree_pdf)[0]
        doc = chunk.to_document()
        assert doc.page_content == chunk.text
        assert doc.metadata["citation"] == chunk.metadata["citation"]


# --------------------------------------------------------------------------- #
# Offline guarantees
# --------------------------------------------------------------------------- #
@pytest.mark.offline
class TestOffline:
    def test_full_ingest_with_sockets_disabled(self, decree_pdf, tmp_path, monkeypatch):
        import socket

        def _blocked(*_a, **_k):
            raise AssertionError("network access during ingestion")

        monkeypatch.setattr(socket, "socket", _blocked)
        monkeypatch.setattr(socket, "create_connection", _blocked)

        chunks = ingest_pdf(decree_pdf)
        path = write_chunks_jsonl(chunks, tmp_path / "chunks.jsonl")
        assert read_chunks_jsonl(path)

    def test_import_pulls_no_network_libs(self):
        code = (
            "import sys, src.ingestion; "
            "bad = {'requests','httpx','aiohttp','urllib.request','openvino',"
            "'torch','transformers','streamlit'} & set(sys.modules); "
            "print(sorted(bad))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=config.BASE_DIR, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "[]", result.stdout
