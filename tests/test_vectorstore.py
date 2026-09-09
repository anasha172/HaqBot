"""Phase 4 — local FAISS vector store: build, query, guardrail score, persist.

No OpenVINO / e5 model is needed: an injected deterministic ``StubEmbedder``
(md5 hashing, stable across processes) and a tiny ``FixedEmbedder`` (exact
vectors) stand in for the real embedder.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time

import numpy as np
import pytest

from src import config
from src.ingestion import Chunk
from src.vectorstore import (
    LegalVectorStore,
    OpenVINOEmbedder,
    RetrievalResult,
    VectorStoreError,
    build_vectorstore,
    l2_normalize,
    load_default_embedder,
    openvino_is_available,
)


# --------------------------------------------------------------------------- #
# Test embedders
# --------------------------------------------------------------------------- #
class StubEmbedder:
    """Deterministic hashing bag-of-words embedder (no model, no network)."""

    id = "stub-hash-embedder"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype="float32")
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.md5(token.encode()).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            vec[idx] += 1.0 if digest[4] & 1 else -1.0
        return vec

    def embed_documents(self, texts):
        return l2_normalize(np.vstack([self._vector(t) for t in texts]))

    def embed_query(self, text):
        return l2_normalize(self._vector(text))


class FixedEmbedder:
    """Returns caller-supplied vectors for exact cosine control."""

    id = "fixed-embedder"

    def __init__(self, table: dict[str, list[float]], dim: int) -> None:
        self.table = table
        self.dim = dim

    def embed_documents(self, texts):
        return np.array([self.table[t] for t in texts], dtype="float32")

    def embed_query(self, text):
        return np.array(self.table[text], dtype="float32")


CORPUS = [
    Chunk(
        "A full-time foreign worker who completes one year of continuous "
        "service is entitled to an end of service gratuity upon termination "
        "of the service.",
        {"article_number": "51", "citation": "[Article 51, Clause 1]",
         "source_title": "Federal Decree-Law No. 33 of 2021"},
    ),
    Chunk(
        "The end of service gratuity is calculated as twenty-one days wage for "
        "each year of the first five years of service.",
        {"article_number": "51", "citation": "[Article 51, Clause 2]",
         "source_title": "Federal Decree-Law No. 33 of 2021"},
    ),
    Chunk(
        "The employer may place the worker on probation for a period not "
        "exceeding six months from the commencement of the work.",
        {"article_number": "9", "citation": "[Article 9, Clause 1]",
         "source_title": "Federal Decree-Law No. 33 of 2021"},
    ),
    Chunk(
        "Employers shall pay wages through the Wage Protection System within "
        "fifteen days from the due date of payment.",
        {"article_number": "5", "citation": "[Article 5, Clause 1]",
         "source_title": "Wage Protection System (WPS) Regulations"},
    ),
    Chunk(
        "Annual leave shall be no less than thirty days for each year of "
        "service completed by the worker.",
        {"article_number": "29", "citation": "[Article 29, Clause 1]",
         "source_title": "Federal Decree-Law No. 33 of 2021"},
    ),
]


@pytest.fixture
def store():
    return LegalVectorStore.build(CORPUS, StubEmbedder())


# --------------------------------------------------------------------------- #
# l2_normalize
# --------------------------------------------------------------------------- #
class TestL2Normalize:
    def test_unit_norm_1d(self):
        out = l2_normalize(np.array([3.0, 4.0]))
        assert np.isclose(np.linalg.norm(out), 1.0)

    def test_unit_norm_rows_2d(self):
        out = l2_normalize(np.array([[3.0, 4.0], [0.0, 2.0]]))
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0)

    def test_zero_vector_is_safe(self):
        assert np.array_equal(l2_normalize(np.zeros(4)), np.zeros(4))
        assert not np.isnan(l2_normalize(np.zeros((2, 3)))).any()


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
class TestBuild:
    def test_index_size_and_dim(self, store):
        assert len(store) == len(CORPUS)
        assert store.index.d == StubEmbedder().dim
        assert store.embedder_id == "stub-hash-embedder"

    def test_build_rejects_empty(self):
        with pytest.raises(VectorStoreError):
            LegalVectorStore.build([], StubEmbedder())

    def test_build_rejects_malformed_matrix(self):
        class BadEmbedder(StubEmbedder):
            def embed_documents(self, texts):
                return np.zeros((len(texts) + 1, self.dim), dtype="float32")

        with pytest.raises(VectorStoreError):
            LegalVectorStore.build(CORPUS, BadEmbedder())


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #
class TestSearch:
    def test_returns_sorted_result(self, store):
        result = store.search("end of service gratuity for a foreign worker", k=3)
        assert isinstance(result, RetrievalResult)
        assert len(result.chunks) == 3
        scores = [c.score for c in result.chunks]
        assert scores == sorted(scores, reverse=True)
        assert [c.rank for c in result.chunks] == [0, 1, 2]
        assert all(-1.0001 <= s <= 1.0001 for s in scores)

    def test_top_hit_is_semantically_closest(self, store):
        result = store.search(
            "how is the end of service gratuity calculated per year of service"
        )
        assert result.chunks[0].metadata["article_number"] == "51"

    def test_k_is_clamped_to_corpus_size(self, store):
        result = store.search("annual leave days", k=99)
        assert len(result.chunks) == len(CORPUS)

    def test_empty_query_raises(self, store):
        with pytest.raises(VectorStoreError):
            store.search("   ")

    def test_search_without_embedder_raises(self, store):
        store.embedder = None
        with pytest.raises(VectorStoreError):
            store.search("anything")

    def test_search_rejects_dim_mismatched_embedder(self, store):
        store.embedder = StubEmbedder(dim=store.dim // 2)
        with pytest.raises(VectorStoreError):
            store.search("anything")

    def test_citations_and_context_block(self, store):
        result = store.search("gratuity", k=2)
        assert result.citations == [c.citation for c in result.chunks]
        block = result.context_block()
        for chunk in result.chunks:
            assert chunk.citation in block
            assert chunk.text in block


# --------------------------------------------------------------------------- #
# Guardrail score (feeds Phase 5)
# --------------------------------------------------------------------------- #
class TestConfidenceScore:
    def _fixed_store(self):
        dim = 3
        table = {
            "doc-a": [1.0, 0.0, 0.0],
            "doc-b": [0.0, 1.0, 0.0],
            "doc-c": [0.0, 0.0, 1.0],
            "close-query": [0.95, 0.31, 0.0],     # cos≈0.95 with doc-a
            "far-query": [1.0, 1.0, 1.0],         # cos≈0.577 with every doc
        }
        chunks = [
            Chunk("doc-a", {"citation": "[Article 1]"}),
            Chunk("doc-b", {"citation": "[Article 2]"}),
            Chunk("doc-c", {"citation": "[Article 3]"}),
        ]
        return LegalVectorStore.build(chunks, FixedEmbedder(table, dim))

    def test_confident_when_top_score_clears_threshold(self):
        result = self._fixed_store().search("close-query", k=3)
        assert result.top_score >= config.SIMILARITY_THRESHOLD
        assert result.is_confident is True

    def test_not_confident_when_all_matches_are_weak(self):
        result = self._fixed_store().search("far-query", k=3)
        assert result.top_score < config.SIMILARITY_THRESHOLD
        assert result.is_confident is False

    def test_empty_result_is_not_confident(self):
        empty = LegalVectorStore.build(
            [Chunk("x", {})], FixedEmbedder({"x": [1.0], "q": [1.0]}, 1)
        )
        empty.index.reset()
        result = empty.search("q")
        assert result.chunks == []
        assert result.is_confident is False


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
class TestPersistence:
    def test_save_writes_all_three_files(self, store, tmp_path):
        out = store.save(tmp_path / "vs")
        assert (out / config.FAISS_INDEX_PATH.name).is_file()
        assert (out / config.VECTORSTORE_DOCS_PATH.name).is_file()
        meta = json.loads((out / config.VECTORSTORE_META_PATH.name).read_text())
        assert meta["count"] == len(CORPUS)
        assert meta["dim"] == store.dim
        assert meta["embedder_id"] == "stub-hash-embedder"
        assert meta["similarity_threshold"] == config.SIMILARITY_THRESHOLD

    def test_roundtrip_preserves_retrieval(self, store, tmp_path):
        store.save(tmp_path / "vs")
        reloaded = LegalVectorStore.load(tmp_path / "vs", StubEmbedder())
        assert len(reloaded) == len(store)
        q = "probation period six months"
        assert (
            reloaded.search(q).chunks[0].text == store.search(q).chunks[0].text
        )

    def test_load_missing_directory_raises(self, tmp_path):
        with pytest.raises(VectorStoreError):
            LegalVectorStore.load(tmp_path / "absent", StubEmbedder())

    def test_load_rejects_dim_mismatch(self, store, tmp_path):
        store.save(tmp_path / "vs")
        with pytest.raises(VectorStoreError):
            LegalVectorStore.load(tmp_path / "vs", StubEmbedder(dim=128))

    def test_load_detects_docstore_index_mismatch(self, store, tmp_path):
        out = store.save(tmp_path / "vs")
        docs = out / config.VECTORSTORE_DOCS_PATH.name
        with docs.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"text": "extra", "metadata": {}}) + "\n")
        with pytest.raises(VectorStoreError):
            LegalVectorStore.load(out, StubEmbedder())

    def test_load_detects_corrupt_docstore_record(self, store, tmp_path):
        out = store.save(tmp_path / "vs")
        (out / config.VECTORSTORE_DOCS_PATH.name).write_text(
            "not-json\n", encoding="utf-8"
        )
        with pytest.raises(VectorStoreError):
            LegalVectorStore.load(out, StubEmbedder())

    def test_load_handles_corrupt_index_file(self, store, tmp_path):
        out = store.save(tmp_path / "vs")
        (out / config.FAISS_INDEX_PATH.name).write_bytes(b"garbage-not-faiss")
        with pytest.raises(VectorStoreError):
            LegalVectorStore.load(out, StubEmbedder())

    def test_load_skips_blank_docstore_lines(self, store, tmp_path):
        out = store.save(tmp_path / "vs")
        docs = out / config.VECTORSTORE_DOCS_PATH.name
        lines = docs.read_text(encoding="utf-8").splitlines()
        docs.write_text(
            lines[0] + "\n\n   \n" + "\n".join(lines[1:]) + "\n", encoding="utf-8"
        )
        reloaded = LegalVectorStore.load(out, StubEmbedder())
        assert len(reloaded) == len(store)

    def test_build_vectorstore_from_jsonl(self, tmp_path, monkeypatch):
        from src import ingestion

        jsonl = tmp_path / "chunks.jsonl"
        ingestion.write_chunks_jsonl(CORPUS, jsonl)
        monkeypatch.setattr(
            "src.vectorstore.load_default_embedder", lambda: StubEmbedder()
        )
        store = build_vectorstore(jsonl, tmp_path / "vs")
        assert len(store) == len(CORPUS)
        assert (tmp_path / "vs" / config.VECTORSTORE_META_PATH.name).is_file()


# --------------------------------------------------------------------------- #
# Retrieval latency (PRD: sub-100 ms)
# --------------------------------------------------------------------------- #
class TestLatency:
    @pytest.fixture
    def big_store(self):
        chunks = [
            Chunk(
                f"Article {i}. The worker shall be entitled to benefit number "
                f"{i} under the labour relations regulation and its clauses.",
                {"article_number": str(i), "citation": f"[Article {i}]"},
            )
            for i in range(1, 401)
        ]
        return LegalVectorStore.build(chunks, StubEmbedder())

    def test_single_query_under_budget(self, big_store):
        # warm up (index / numpy paths)
        big_store.search("worker entitlement benefit clause")
        samples = []
        for _ in range(20):
            t0 = time.perf_counter()
            result = big_store.search("worker entitlement benefit clause", k=4)
            samples.append((time.perf_counter() - t0) * 1000.0)
        median_ms = sorted(samples)[len(samples) // 2]
        assert median_ms < config.RETRIEVAL_TIMING_BUDGET_MS, (
            f"median retrieval {median_ms:.2f} ms exceeds "
            f"{config.RETRIEVAL_TIMING_BUDGET_MS} ms budget"
        )
        assert result.elapsed_ms < config.RETRIEVAL_TIMING_BUDGET_MS


# --------------------------------------------------------------------------- #
# OpenVINO embedder guards (no model present in CI)
# --------------------------------------------------------------------------- #
class TestOpenVINOEmbedder:
    def test_missing_model_dir_raises_clean_error(self, tmp_path):
        with pytest.raises(VectorStoreError):
            OpenVINOEmbedder(tmp_path / "nope")

    def test_load_default_embedder_surfaces_clean_error(self):
        if openvino_is_available() and config.EMBEDDING_MODEL_DIR.is_dir():
            pytest.skip("real model present — nothing to assert here")
        with pytest.raises(VectorStoreError):
            load_default_embedder()

    def test_mean_pooling_and_prefixes_with_fake_backend(self, tmp_path, monkeypatch):
        """Exercise the real pooling / normalisation path with stub OV + tokenizer."""
        import types

        model_dir = tmp_path / "e5-small-ov"
        model_dir.mkdir()

        class _FakeTokenizer:
            @classmethod
            def from_pretrained(cls, *a, **k):
                return cls()

            def __call__(self, texts, **kw):
                self.last_texts = list(texts)
                seq = 4
                mask = np.ones((len(texts), seq), dtype="int64")
                mask[:, -1] = 0  # final token is padding
                return {"attention_mask": mask}

        class _FakeOut:
            def __init__(self, lhs):
                self.last_hidden_state = lhs

        class _FakeModel:
            config = types.SimpleNamespace(hidden_size=8)

            @classmethod
            def from_pretrained(cls, *a, **k):
                return cls()

            def __call__(self, **enc):
                n, seq = enc["attention_mask"].shape
                return _FakeOut(np.ones((n, seq, 8), dtype="float32"))

        fake_optimum_intel = types.ModuleType("optimum.intel")
        fake_optimum_intel.OVModelForFeatureExtraction = _FakeModel
        monkeypatch.setitem(sys.modules, "optimum", types.ModuleType("optimum"))
        monkeypatch.setitem(sys.modules, "optimum.intel", fake_optimum_intel)
        monkeypatch.setattr("transformers.AutoTokenizer", _FakeTokenizer, raising=False)
        monkeypatch.setattr("src.vectorstore.openvino_is_available", lambda: True)

        emb = OpenVINOEmbedder(model_dir)
        assert emb.dim == 8

        q = emb.embed_query("gratuity")
        assert q.shape == (8,)
        assert np.isclose(np.linalg.norm(q), 1.0)
        assert emb._tokenizer.last_texts == [config.E5_QUERY_PREFIX + "gratuity"]

        docs = emb.embed_documents(["one", "two"])
        assert docs.shape == (2, 8)
        assert np.allclose(np.linalg.norm(docs, axis=1), 1.0)
        assert emb._tokenizer.last_texts[0].startswith(config.E5_PASSAGE_PREFIX)


# --------------------------------------------------------------------------- #
# Offline guarantees
# --------------------------------------------------------------------------- #
@pytest.mark.offline
class TestOffline:
    def test_full_cycle_with_sockets_disabled(self, tmp_path, monkeypatch):
        import socket

        def _blocked(*_a, **_k):
            raise AssertionError("network access during vector store use")

        monkeypatch.setattr(socket, "socket", _blocked)
        monkeypatch.setattr(socket, "create_connection", _blocked)

        vs = LegalVectorStore.build(CORPUS, StubEmbedder())
        out = vs.save(tmp_path / "vs")
        again = LegalVectorStore.load(out, StubEmbedder())
        assert again.search("gratuity").chunks

    def test_import_pulls_no_network_or_ml_libs(self):
        code = (
            "import sys, src.vectorstore; "
            "bad = {'requests','httpx','aiohttp','urllib.request','openvino',"
            "'optimum','torch','transformers','streamlit'} & set(sys.modules); "
            "print(sorted(bad))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=config.BASE_DIR, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "[]", result.stdout
