"""HaqBot — local FAISS vector store over cited legal chunks.

Cosine-similarity retrieval with **zero network access**. The FAISS
``IndexFlatIP`` is held fully in RAM; embeddings are L2-normalised so an inner
product equals cosine similarity in ``[-1, 1]`` (higher == more relevant),
matching :data:`config.SIMILARITY_THRESHOLD`.

The embedding model is pluggable via the :class:`Embedder` protocol:
  * :class:`OpenVINOEmbedder` — the production INT8 ``multilingual-e5-small``
    OpenVINO IR loaded from ``models/e5-small-ov`` (lazy heavy imports).
  * any object implementing ``embed_documents`` / ``embed_query`` — used by tests
    and offline development.
"""

from __future__ import annotations

import importlib.util
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Sequence, runtime_checkable

import numpy as np

from src import config
from src.ingestion import Chunk

if TYPE_CHECKING:  # pragma: no cover
    import faiss

__all__ = [
    "VectorStoreError",
    "Embedder",
    "OpenVINOEmbedder",
    "RetrievedChunk",
    "RetrievalResult",
    "LegalVectorStore",
    "l2_normalize",
    "openvino_is_available",
    "load_default_embedder",
    "build_vectorstore",
]

_SCHEMA_VERSION = 1


class VectorStoreError(Exception):
    """Any failure building, loading, or querying the local index."""


# =========================================================================== #
# Embedder contract
# =========================================================================== #
@runtime_checkable
class Embedder(Protocol):
    """Minimal embedding interface. Implementations must L2-normalise output."""

    id: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise a vector (1-D) or each row of a matrix (2-D). Zero-safe."""
    arr = np.asarray(matrix, dtype="float32")
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm else arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return arr / norms


def openvino_is_available() -> bool:
    """True only if both ``openvino`` and ``optimum`` are importable."""
    return (
        importlib.util.find_spec("openvino") is not None
        and importlib.util.find_spec("optimum") is not None
    )


# =========================================================================== #
# Production embedder (OpenVINO INT8 multilingual-e5-small)
# =========================================================================== #
class OpenVINOEmbedder:
    """INT8 ``multilingual-e5-small`` served by OpenVINO, mean-pooled + normed."""

    id = "multilingual-e5-small-ov-int8"

    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        device: str | None = None,
        max_tokens: int | None = None,
    ) -> None:
        directory = Path(model_dir or config.EMBEDDING_MODEL_DIR)
        if not directory.is_dir():
            raise VectorStoreError(
                f"Embedding model not found at {directory}. Run "
                f"`python -m src.quantization embedding` on a build machine."
            )
        if not openvino_is_available():
            raise VectorStoreError(
                "openvino / optimum-intel are not installed in this environment."
            )
        from optimum.intel import OVModelForFeatureExtraction  # heavy, lazy
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(directory)
        self._model = OVModelForFeatureExtraction.from_pretrained(
            directory, device=device or config.OPENVINO_DEVICE
        )
        self.max_tokens = int(max_tokens or config.EMBEDDING_MAX_TOKENS)
        self.dim = int(getattr(self._model.config, "hidden_size", config.EMBEDDING_DIM))

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        enc = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="np",
        )
        outputs = self._model(**enc)
        hidden = np.asarray(outputs.last_hidden_state, dtype="float32")
        mask = np.asarray(enc["attention_mask"], dtype="float32")[..., None]
        summed = (hidden * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        return l2_normalize(summed / counts)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([config.E5_PASSAGE_PREFIX + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([config.E5_QUERY_PREFIX + text])[0]


def load_default_embedder() -> Embedder:
    """Return the production :class:`OpenVINOEmbedder`."""
    return OpenVINOEmbedder()


# =========================================================================== #
# Retrieval results
# =========================================================================== #
@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    metadata: dict[str, Any]
    score: float
    rank: int

    @property
    def citation(self) -> str:
        return self.metadata.get("citation", "")


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]
    elapsed_ms: float

    @property
    def top_score(self) -> float:
        return self.chunks[0].score if self.chunks else -1.0

    @property
    def is_confident(self) -> bool:
        """True when the best match clears the anti-hallucination threshold."""
        return self.top_score >= config.SIMILARITY_THRESHOLD

    @property
    def citations(self) -> list[str]:
        seen: list[str] = []
        for chunk in self.chunks:
            if chunk.citation and chunk.citation not in seen:
                seen.append(chunk.citation)
        return seen

    def context_block(self) -> str:
        """Formatted ``{context}`` string for the grounded LLM prompt."""
        return "\n\n".join(
            f"{c.citation} {c.text}".strip() for c in self.chunks
        )


# =========================================================================== #
# The store
# =========================================================================== #
class LegalVectorStore:
    """In-RAM FAISS flat cosine index plus parallel text / metadata arrays."""

    def __init__(
        self,
        index: "faiss.Index",
        texts: list[str],
        metadatas: list[dict[str, Any]],
        *,
        embedder: Embedder | None,
        dim: int,
        embedder_id: str,
    ) -> None:
        self.index = index
        self.texts = texts
        self.metadatas = metadatas
        self.embedder = embedder
        self.dim = dim
        self.embedder_id = embedder_id

    def __len__(self) -> int:
        return int(self.index.ntotal)

    # -- build ---------------------------------------------------------------
    @classmethod
    def build(
        cls, chunks: Sequence[Chunk], embedder: Embedder
    ) -> "LegalVectorStore":
        if not chunks:
            raise VectorStoreError("Cannot build an index from zero chunks.")
        import faiss

        texts = [c.text for c in chunks]
        metadatas = [dict(c.metadata) for c in chunks]
        vectors = l2_normalize(embedder.embed_documents(texts)).astype("float32")
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise VectorStoreError("Embedder returned a malformed matrix.")
        dim = int(vectors.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        return cls(
            index,
            texts,
            metadatas,
            embedder=embedder,
            dim=dim,
            embedder_id=getattr(embedder, "id", "unknown"),
        )

    # -- query -------------------------------------------------------------
    def search(self, query: str, k: int | None = None) -> RetrievalResult:
        if not query or not query.strip():
            raise VectorStoreError("Query text is empty.")
        if self.embedder is None:
            raise VectorStoreError("No embedder bound to this store.")
        if self.index.ntotal == 0:
            return RetrievalResult(query=query, chunks=[], elapsed_ms=0.0)

        top_k = min(k or config.RETRIEVAL_TOP_K, int(self.index.ntotal))
        start = time.perf_counter()
        q_vec = (
            l2_normalize(self.embedder.embed_query(query))
            .astype("float32")
            .reshape(1, -1)
        )
        if q_vec.shape[1] != self.dim:
            raise VectorStoreError(
                f"Query embedding dim {q_vec.shape[1]} != index dim {self.dim}."
            )
        scores, indices = self.index.search(q_vec, top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        results: list[RetrievedChunk] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue
            results.append(
                RetrievedChunk(
                    text=self.texts[idx],
                    metadata=self.metadatas[idx],
                    score=float(score),
                    rank=rank,
                )
            )
        return RetrievalResult(query=query, chunks=results, elapsed_ms=elapsed_ms)

    # -- persistence -----------------------------------------------------------
    def save(self, directory: str | Path | None = None) -> Path:
        import faiss

        out_dir = Path(directory or config.VECTORSTORE_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out_dir / config.FAISS_INDEX_PATH.name))
        with (out_dir / config.VECTORSTORE_DOCS_PATH.name).open(
            "w", encoding="utf-8"
        ) as fh:
            for text, meta in zip(self.texts, self.metadatas):
                fh.write(
                    json.dumps(
                        {"text": text, "metadata": meta}, ensure_ascii=False
                    )
                    + "\n"
                )
        (out_dir / config.VECTORSTORE_META_PATH.name).write_text(
            json.dumps(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "dim": self.dim,
                    "count": len(self.texts),
                    "embedder_id": self.embedder_id,
                    "similarity_threshold": config.SIMILARITY_THRESHOLD,
                    "created_at": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return out_dir

    @classmethod
    def load(
        cls,
        directory: str | Path | None = None,
        embedder: Embedder | None = None,
    ) -> "LegalVectorStore":
        import faiss

        in_dir = Path(directory or config.VECTORSTORE_DIR)
        index_path = in_dir / config.FAISS_INDEX_PATH.name
        docs_path = in_dir / config.VECTORSTORE_DOCS_PATH.name
        meta_path = in_dir / config.VECTORSTORE_META_PATH.name
        for path in (index_path, docs_path, meta_path):
            if not path.is_file():
                raise VectorStoreError(f"Vector store file missing: {path}")

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            index = faiss.read_index(str(index_path))
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Corrupt vector store in {in_dir}: {exc}") from exc

        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []
        with docs_path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    texts.append(record["text"])
                    metadatas.append(record["metadata"])
                except (json.JSONDecodeError, KeyError) as exc:
                    raise VectorStoreError(
                        f"Corrupt docstore record at line {line_no}."
                    ) from exc

        dim = int(meta.get("dim", index.d))
        if index.d != dim or int(index.ntotal) != len(texts):
            raise VectorStoreError(
                "Vector store files are inconsistent "
                f"(index d={index.d}, ntotal={index.ntotal}, docs={len(texts)})."
            )
        if embedder is not None and getattr(embedder, "dim", dim) != dim:
            raise VectorStoreError(
                f"Embedder dim {embedder.dim} != stored index dim {dim}."
            )
        return cls(
            index,
            texts,
            metadatas,
            embedder=embedder,
            dim=dim,
            embedder_id=str(meta.get("embedder_id", "unknown")),
        )


# =========================================================================== #
# Orchestration + CLI
# =========================================================================== #
def build_vectorstore(
    chunks_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    *,
    embedder: Embedder | None = None,
) -> LegalVectorStore:
    """Read chunks JSONL -> embed -> build FAISS index -> persist to disk."""
    from src.ingestion import read_chunks_jsonl

    chunks = read_chunks_jsonl(chunks_path)
    store = LegalVectorStore.build(chunks, embedder or load_default_embedder())
    store.save(out_dir)
    return store


def _main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI
    import argparse

    parser = argparse.ArgumentParser(
        description="Build the local FAISS legal vector store."
    )
    parser.add_argument("chunks", nargs="?", default=str(config.CHUNKS_META_PATH))
    parser.add_argument("out", nargs="?", default=str(config.VECTORSTORE_DIR))
    args = parser.parse_args(argv)
    try:
        store = build_vectorstore(args.chunks, args.out)
    except (VectorStoreError, Exception) as exc:  # noqa: BLE001
        print(f"vector store build failed: {exc}")
        return 1
    print(f"indexed {len(store)} chunks (dim {store.dim}) -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
