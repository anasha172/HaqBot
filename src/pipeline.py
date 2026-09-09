"""HaqBot — grounded RAG pipeline with a hard anti-hallucination guardrail.

    query
      -> LegalVectorStore.search  (cosine over INT8 e5 embeddings)
      -> GUARDRAIL: if best cosine < SIMILARITY_THRESHOLD (0.65) or no hits,
         the LLM is NEVER called — return the exact MOHRE referral fallback
      -> otherwise: grounded, citation-enforced prompt -> OpenVINO Qwen2.5
      -> post-process: strip echoed disclaimer, enforce >=1 citation, detect a
         parroted refusal, append the localised disclaimer once
      -> Answer (text, citations, sources, used_fallback, top_score, language)

The LLM is pluggable: any object exposing ``invoke(prompt: str) -> str``
(the LangChain ``Runnable`` surface) works. :class:`OpenVINOChatLLM` is the
production implementation and also exposes ``as_langchain()``.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from src import config
from src.vectorstore import (
    Embedder,
    LegalVectorStore,
    RetrievalResult,
    RetrievedChunk,
    VectorStoreError,
)

__all__ = [
    "PipelineError",
    "LLMProtocol",
    "OpenVINOChatLLM",
    "Answer",
    "HaqBotPipeline",
    "extract_citations",
    "resolve_language",
    "build_default_pipeline",
]

_CITATION_RE = re.compile(r"\[[^\]\n]{1,120}\]")
_ARTICLE_CITATION_RE = re.compile(r"\[\s*Article\s+\d+[^\]]*\]", re.IGNORECASE)
_DISCLAIMER_LINE_RE = re.compile(
    r"(?im)^\s*(disclaimer|إخلاء مسؤولية|अस्वीकरण|دستبرداری|നിരാകരണം)\b.*$"
)


class PipelineError(Exception):
    """Any unrecoverable problem answering a query (surfaced as safe text)."""


# =========================================================================== #
# LLM contract + production implementation
# =========================================================================== #
@runtime_checkable
class LLMProtocol(Protocol):
    def invoke(self, prompt: str) -> str: ...


class OpenVINOChatLLM:
    """Qwen2.5-Instruct served by OpenVINO (INT8). Heavy imports are lazy."""

    id = "qwen2.5-1.5b-instruct-ov-int8"

    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        device: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> None:
        directory = Path(model_dir or config.LLM_MODEL_DIR)
        if not directory.is_dir():
            raise PipelineError(
                f"LLM not found at {directory}. Run "
                f"`python -m src.quantization llm` on a build machine."
            )
        try:
            from optimum.intel import OVModelForCausalLM  # heavy, lazy
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise PipelineError(
                "openvino / optimum-intel are not installed in this environment."
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(directory)
        self._model = OVModelForCausalLM.from_pretrained(
            directory, device=device or config.OPENVINO_DEVICE
        )
        self.max_new_tokens = int(max_new_tokens or config.MAX_NEW_TOKENS)
        self.temperature = (
            config.TEMPERATURE if temperature is None else float(temperature)
        )

    def invoke(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt")
        do_sample = self.temperature > 0.0
        generated = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature=self.temperature if do_sample else None,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        new_tokens = generated[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()

    def as_langchain(self):
        """Expose this LLM as a LangChain ``Runnable`` (lazy import)."""
        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(self.invoke)


# =========================================================================== #
# Helpers
# =========================================================================== #
def resolve_language(language: str | None) -> str:
    """Return a supported language code, defaulting to English."""
    code = (language or "").strip().lower()
    return code if code in config.SUPPORTED_LANGUAGES else config.DEFAULT_LANGUAGE


def extract_citations(text: str) -> list[str]:
    """Ordered, de-duplicated ``[...]`` citations found in ``text``."""
    seen: list[str] = []
    for match in _CITATION_RE.findall(text or ""):
        cleaned = re.sub(r"\s+", " ", match).strip()
        if cleaned not in seen:
            seen.append(cleaned)
    return seen


def _strip_model_disclaimer(text: str) -> str:
    """Remove any disclaimer line the model added — we append our canonical one."""
    return _DISCLAIMER_LINE_RE.sub("", text or "").rstrip()


def _looks_like_refusal(text: str) -> bool:
    return config.REFUSAL_MARKER.lower() in (text or "").lower()


# =========================================================================== #
# Result type
# =========================================================================== #
@dataclass(frozen=True)
class Answer:
    text: str
    language: str
    citations: list[str]
    sources: list[RetrievedChunk]
    used_fallback: bool
    top_score: float
    elapsed_ms: float
    disclaimer: str
    unsupported_citations: list[str] = field(default_factory=list)

    @property
    def source_documents(self) -> list[str]:
        seen: list[str] = []
        for chunk in self.sources:
            title = chunk.metadata.get("source_title") or chunk.metadata.get(
                "doc_title", ""
            )
            if title and title not in seen:
                seen.append(title)
        return seen

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "citations": list(self.citations),
            "used_fallback": self.used_fallback,
            "top_score": self.top_score,
            "elapsed_ms": self.elapsed_ms,
            "disclaimer": self.disclaimer,
            "unsupported_citations": list(self.unsupported_citations),
            "sources": [
                {
                    "text": c.text,
                    "score": c.score,
                    "citation": c.citation,
                    "metadata": c.metadata,
                }
                for c in self.sources
            ],
        }


# =========================================================================== #
# Pipeline
# =========================================================================== #
class HaqBotPipeline:
    """Retrieval + guardrail + grounded generation."""

    def __init__(
        self,
        vectorstore: LegalVectorStore,
        llm: LLMProtocol,
        *,
        top_k: int | None = None,
    ) -> None:
        if not hasattr(llm, "invoke"):
            raise PipelineError("llm must expose an invoke(prompt) -> str method.")
        self.vectorstore = vectorstore
        self.llm = llm
        self.top_k = int(top_k or config.RETRIEVAL_TOP_K)

    # -- prompt -----------------------------------------------------------------
    def build_prompt(self, query: str, result: RetrievalResult, language: str) -> str:
        context = result.context_block()
        if len(context) > config.MAX_CONTEXT_CHARS:
            context = context[: config.MAX_CONTEXT_CHARS].rsplit("\n\n", 1)[0]
        grounded = config.SYSTEM_PROMPT.format(context=context, query=query)
        instruction = config.LANGUAGE_INSTRUCTIONS.get(
            language, config.LANGUAGE_INSTRUCTIONS[config.DEFAULT_LANGUAGE]
        )
        return f"{grounded}\n\n{instruction}"

    # -- fallback -------------------------------------------------------------
    def _fallback_answer(
        self,
        result: RetrievalResult | None,
        language: str,
        started: float,
    ) -> Answer:
        return Answer(
            text=config.LOCALIZED_FALLBACK.get(
                language, config.FALLBACK_MESSAGE
            ),
            language=language,
            citations=[],
            sources=list(result.chunks) if result else [],
            used_fallback=True,
            top_score=result.top_score if result else -1.0,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            disclaimer=config.LOCALIZED_DISCLAIMER.get(
                language, config.LEGAL_DISCLAIMER
            ),
        )

    # -- main entry point --------------------------------------------------
    def answer(
        self, query: str, *, language: str | None = None, k: int | None = None
    ) -> Answer:
        started = time.perf_counter()
        language = resolve_language(language)

        if query is None or not query.strip():
            raise PipelineError("Please type a question.")
        query = query.strip()
        if len(query) > config.MAX_QUERY_CHARS:
            query = query[: config.MAX_QUERY_CHARS].rstrip()

        try:
            result = self.vectorstore.search(query, k or self.top_k)
        except VectorStoreError as exc:
            raise PipelineError(f"Could not search the legal database: {exc}") from exc

        # ---- ANTI-HALLUCINATION GUARDRAIL -----------------------------------
        if not result.chunks or not result.is_confident:
            return self._fallback_answer(result, language, started)

        # ---- grounded generation ------------------------------------------
        prompt = self.build_prompt(query, result, language)
        try:
            raw = self.llm.invoke(prompt)
        except Exception as exc:  # noqa: BLE001 - never leak a traceback
            raise PipelineError(
                f"The offline model failed to generate an answer: {exc}"
            ) from exc

        raw = (raw or "").strip()
        if not raw or _looks_like_refusal(raw):
            return self._fallback_answer(result, language, started)

        text = _strip_model_disclaimer(raw)
        retrieved_citations = result.citations

        # citation enforcement: every answer must carry a bracketed citation
        citations = extract_citations(text)
        if not _ARTICLE_CITATION_RE.search(text):
            if retrieved_citations:
                text = f"{text}\n\nSources: " + ", ".join(retrieved_citations)
                citations = extract_citations(text)
            else:  # pragma: no cover - guardrail already blocks this path
                return self._fallback_answer(result, language, started)

        allowed_articles = {
            str(c.metadata.get("article_number"))
            for c in result.chunks
            if c.metadata.get("article_number")
        }
        unsupported = [
            cit
            for cit in citations
            for m in [re.search(r"Article\s+(\d+)", cit, re.IGNORECASE)]
            if m and m.group(1) not in allowed_articles
        ]

        disclaimer = config.LOCALIZED_DISCLAIMER.get(
            language, config.LEGAL_DISCLAIMER
        )
        return Answer(
            text=text.strip(),
            language=language,
            citations=citations,
            sources=list(result.chunks),
            used_fallback=False,
            top_score=result.top_score,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            disclaimer=disclaimer,
            unsupported_citations=unsupported,
        )


# =========================================================================== #
# Factory + CLI
# =========================================================================== #
def build_default_pipeline(
    *,
    index_dir: str | Path | None = None,
    embedder: Embedder | None = None,
    llm: LLMProtocol | None = None,
) -> HaqBotPipeline:
    """Load the on-disk vector store + OpenVINO embedder + OpenVINO LLM."""
    from src.vectorstore import load_default_embedder

    emb = embedder or load_default_embedder()
    store = LegalVectorStore.load(index_dir, emb)
    return HaqBotPipeline(store, llm or OpenVINOChatLLM())


def _main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI
    import argparse

    parser = argparse.ArgumentParser(description="Ask HaqBot a question offline.")
    parser.add_argument("question")
    parser.add_argument("--language", default="en", choices=list(config.SUPPORTED_LANGUAGES))
    args = parser.parse_args(argv)
    try:
        pipe = build_default_pipeline()
        answer = pipe.answer(args.question, language=args.language)
    except PipelineError as exc:
        print(f"HaqBot: {exc}")
        return 1
    print(answer.text)
    print()
    print(answer.disclaimer)
    if answer.used_fallback:
        print("\n[no confident match — MOHRE referral]")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
