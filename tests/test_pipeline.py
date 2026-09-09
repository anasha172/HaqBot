"""Phase 5 — grounded RAG pipeline & anti-hallucination guardrail.

No OpenVINO model is used: a deterministic ``FixedEmbedder`` gives exact control
of the retrieval cosine (so the 0.65 guardrail can be tested precisely), and a
``StubLLM`` records prompts / returns canned text.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from src import config
from src.ingestion import Chunk
from src.pipeline import (
    Answer,
    HaqBotPipeline,
    OpenVINOChatLLM,
    PipelineError,
    extract_citations,
    resolve_language,
)
from src.vectorstore import LegalVectorStore


# --------------------------------------------------------------------------- #
# Doubles
# --------------------------------------------------------------------------- #
class StubLLM:
    def __init__(self, response="Under [Article 51, Clause 1] the worker is "
                                "entitled to an end of service gratuity."):
        self.response = response
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class RaisingLLM:
    def invoke(self, prompt: str) -> str:
        raise RuntimeError("model exploded")


class FixedEmbedder:
    id = "fixed-embedder"

    def __init__(self, table, dim):
        self.table = table
        self.dim = dim

    def embed_documents(self, texts):
        return np.array([self.table[t] for t in texts], dtype="float32")

    def embed_query(self, text):
        return np.array(self.table[text], dtype="float32")


DOCS = [
    Chunk("doc-a gratuity", {
        "article_number": "51", "citation": "[Article 51, Clause 1]",
        "source_title": "Federal Decree-Law No. 33 of 2021"}),
    Chunk("doc-b probation", {
        "article_number": "9", "citation": "[Article 9, Clause 1]",
        "source_title": "Federal Decree-Law No. 33 of 2021"}),
    Chunk("doc-c wages", {
        "article_number": "5", "citation": "[Article 5, Clause 1]",
        "source_title": "Wage Protection System (WPS) Regulations"}),
]

TABLE = {
    "doc-a gratuity": [1.0, 0.0, 0.0],
    "doc-b probation": [0.0, 1.0, 0.0],
    "doc-c wages": [0.0, 0.0, 1.0],
    "confident query": [0.95, 0.31, 0.0],   # cos ~0.95 with doc-a
    "weak query": [1.0, 1.0, 1.0],          # cos ~0.577 with every doc
}


def make_pipeline(llm=None, *, table=None):
    store = LegalVectorStore.build(DOCS, FixedEmbedder(table or TABLE, 3))
    return HaqBotPipeline(store, llm or StubLLM())


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class TestHelpers:
    def test_resolve_language(self):
        assert resolve_language("HI") == "hi"
        assert resolve_language("fr") == "en"
        assert resolve_language(None) == "en"
        assert resolve_language("  ur ") == "ur"

    def test_extract_citations(self):
        text = "See [Article 9, Clause 1] and [Article 9, Clause 1] and [WPS]."
        assert extract_citations(text) == ["[Article 9, Clause 1]", "[WPS]"]
        assert extract_citations("no brackets here") == []


# --------------------------------------------------------------------------- #
# The guardrail
# --------------------------------------------------------------------------- #
class TestGuardrail:
    def test_confident_match_invokes_llm(self):
        llm = StubLLM()
        pipe = make_pipeline(llm)
        answer = pipe.answer("confident query")
        assert answer.used_fallback is False
        assert answer.top_score >= config.SIMILARITY_THRESHOLD
        assert llm.prompts, "LLM should have been called"

    def test_weak_match_skips_llm_and_returns_fallback(self):
        llm = StubLLM()
        pipe = make_pipeline(llm)
        answer = pipe.answer("weak query")
        assert answer.used_fallback is True
        assert answer.top_score < config.SIMILARITY_THRESHOLD
        assert llm.prompts == [], "LLM must NOT be called below threshold"
        assert answer.text == config.LOCALIZED_FALLBACK["en"]
        assert "80084" in answer.text
        assert len(answer.sources) == 3  # weak matches still shown to the user
        assert answer.citations == []

    def test_empty_index_returns_fallback(self):
        store = LegalVectorStore.build(DOCS, FixedEmbedder(TABLE, 3))
        store.index.reset()
        llm = StubLLM()
        answer = HaqBotPipeline(store, llm).answer("confident query")
        assert answer.used_fallback is True
        assert llm.prompts == []

    def test_threshold_uses_config_value(self):
        # Top cosine with doc-a is ~0.640 (just under) vs ~0.660 (just over);
        # the two spread components stay well below threshold either way.
        table = dict(TABLE)
        table["just under"] = [1.0, 0.849, 0.849]   # max cos ~0.640 with doc-a
        table["just over"] = [1.0, 0.805, 0.805]    # max cos ~0.660 with doc-a
        pipe = make_pipeline(StubLLM(), table=table)
        assert pipe.answer("just under").used_fallback is True
        assert pipe.answer("just over").used_fallback is False


# --------------------------------------------------------------------------- #
# Grounded generation + prompt
# --------------------------------------------------------------------------- #
class TestGeneration:
    def test_prompt_contains_context_query_and_language_rule(self):
        llm = StubLLM()
        make_pipeline(llm).answer("confident query", language="hi")
        prompt = llm.prompts[0]
        assert "CRITICAL INSTRUCTIONS" in prompt
        assert "[Article X, Clause Y]" in prompt
        assert "[Article 51, Clause 1] doc-a gratuity" in prompt   # context block
        assert "confident query" in prompt
        assert config.LANGUAGE_INSTRUCTIONS["hi"] in prompt

    def test_grounded_answer_keeps_model_citation(self):
        answer = make_pipeline(StubLLM()).answer("confident query")
        assert answer.used_fallback is False
        assert "[Article 51, Clause 1]" in answer.citations
        assert "Sources:" not in answer.text

    def test_model_disclaimer_line_is_stripped(self):
        llm = StubLLM(
            "The worker is entitled under [Article 51, Clause 1].\n"
            "Disclaimer: this is not legal advice and blah blah."
        )
        answer = make_pipeline(llm).answer("confident query")
        assert "not legal advice and blah" not in answer.text
        assert answer.disclaimer == config.LOCALIZED_DISCLAIMER["en"]

    def test_llm_failure_becomes_pipeline_error(self):
        with pytest.raises(PipelineError):
            make_pipeline(RaisingLLM()).answer("confident query")

    def test_long_context_is_truncated_before_prompting(self):
        big = "doc-a " + "gratuity clause text " * 500          # ~10k chars
        docs = [Chunk(big, {"article_number": "51",
                            "citation": "[Article 51, Clause 1]",
                            "source_title": "Federal Decree-Law No. 33 of 2021"})]
        table = {big: [1.0, 0.0, 0.0], "confident query": [0.95, 0.31, 0.0]}
        store = LegalVectorStore.build(docs, FixedEmbedder(table, 3))
        llm = StubLLM()
        HaqBotPipeline(store, llm).answer("confident query")
        assert len(llm.prompts[0]) <= (
            config.MAX_CONTEXT_CHARS + len(config.SYSTEM_PROMPT) + 400
        )


# --------------------------------------------------------------------------- #
# Citation enforcement
# --------------------------------------------------------------------------- #
class TestCitationEnforcement:
    def test_missing_citation_gets_sources_appended(self):
        llm = StubLLM("A worker is entitled to a gratuity after one year.")
        answer = make_pipeline(llm).answer("confident query")
        assert answer.used_fallback is False
        assert "Sources:" in answer.text
        assert "[Article 51, Clause 1]" in answer.text
        assert answer.citations

    def test_citation_outside_retrieved_context_is_flagged(self):
        llm = StubLLM("This is covered by [Article 99, Clause 3] somewhere.")
        answer = make_pipeline(llm).answer("confident query")
        assert "[Article 99, Clause 3]" in answer.unsupported_citations

    def test_retrieved_citation_is_not_flagged(self):
        answer = make_pipeline(StubLLM()).answer("confident query")
        assert answer.unsupported_citations == []


# --------------------------------------------------------------------------- #
# Refusal / empty model output
# --------------------------------------------------------------------------- #
class TestRefusalDetection:
    def test_model_parroting_the_refusal_is_treated_as_fallback(self):
        answer = make_pipeline(StubLLM(config.FALLBACK_MESSAGE)).answer(
            "confident query"
        )
        assert answer.used_fallback is True
        assert answer.text == config.LOCALIZED_FALLBACK["en"]

    def test_empty_model_output_is_treated_as_fallback(self):
        answer = make_pipeline(StubLLM("   ")).answer("confident query")
        assert answer.used_fallback is True


# --------------------------------------------------------------------------- #
# Multilingual
# --------------------------------------------------------------------------- #
class TestMultilingual:
    @pytest.mark.parametrize("lang", ["en", "hi", "ur", "ml", "ar"])
    def test_every_language_has_prompt_fallback_and_disclaimer(self, lang):
        assert lang in config.LANGUAGE_INSTRUCTIONS
        assert "80084" in config.LOCALIZED_FALLBACK[lang]
        assert lang in config.LOCALIZED_DISCLAIMER

    def test_confident_answer_localises_disclaimer(self):
        answer = make_pipeline(StubLLM()).answer("confident query", language="ml")
        assert answer.language == "ml"
        assert answer.disclaimer == config.LOCALIZED_DISCLAIMER["ml"]

    def test_fallback_is_localised(self):
        answer = make_pipeline(StubLLM()).answer("weak query", language="ur")
        assert answer.text == config.LOCALIZED_FALLBACK["ur"]
        assert "80084" in answer.text

    def test_unknown_language_falls_back_to_english(self):
        answer = make_pipeline(StubLLM()).answer("confident query", language="fr")
        assert answer.language == "en"


# --------------------------------------------------------------------------- #
# Input handling / robustness
# --------------------------------------------------------------------------- #
class TestInputHandling:
    @pytest.mark.parametrize("bad", ["", "   ", "\n\t", None])
    def test_empty_query_raises_pipeline_error(self, bad):
        with pytest.raises(PipelineError):
            make_pipeline(StubLLM()).answer(bad)

    def test_overlong_query_is_truncated_not_rejected(self):
        llm = StubLLM()
        long_q = "confident query " + "x" * 5000
        table = dict(TABLE)
        table[long_q[: config.MAX_QUERY_CHARS].rstrip()] = [0.95, 0.31, 0.0]
        pipe = make_pipeline(llm, table=table)
        answer = pipe.answer(long_q)
        assert answer.used_fallback is False
        assert len(llm.prompts[0]) < len(long_q) + len(config.SYSTEM_PROMPT) + 500

    def test_llm_without_invoke_is_rejected_at_construction(self):
        store = LegalVectorStore.build(DOCS, FixedEmbedder(TABLE, 3))
        with pytest.raises(PipelineError):
            HaqBotPipeline(store, object())

    def test_vectorstore_error_becomes_pipeline_error(self, monkeypatch):
        pipe = make_pipeline(StubLLM())

        def _boom(*_a, **_k):
            from src.vectorstore import VectorStoreError

            raise VectorStoreError("index unreadable")

        monkeypatch.setattr(pipe.vectorstore, "search", _boom)
        with pytest.raises(PipelineError):
            pipe.answer("confident query")


# --------------------------------------------------------------------------- #
# Answer shape
# --------------------------------------------------------------------------- #
class TestAnswer:
    def test_to_dict_and_source_documents(self):
        answer = make_pipeline(StubLLM()).answer("confident query")
        data = answer.to_dict()
        assert set(data) >= {
            "text", "language", "citations", "used_fallback", "top_score",
            "elapsed_ms", "disclaimer", "sources", "unsupported_citations",
        }
        assert data["sources"][0]["citation"] == "[Article 51, Clause 1]"
        assert answer.source_documents[0] == "Federal Decree-Law No. 33 of 2021"
        assert "Wage Protection System (WPS) Regulations" in answer.source_documents
        assert len(answer.source_documents) == len(set(answer.source_documents))
        assert answer.elapsed_ms >= 0.0


# --------------------------------------------------------------------------- #
# OpenVINO LLM wrapper
# --------------------------------------------------------------------------- #
class TestOpenVINOChatLLM:
    def test_missing_model_dir_raises_clean_error(self, tmp_path):
        with pytest.raises(PipelineError):
            OpenVINOChatLLM(tmp_path / "nope")

    def test_invoke_with_fake_backend(self, tmp_path, monkeypatch):
        import types

        model_dir = tmp_path / "qwen-ov"
        model_dir.mkdir()

        class _FakeTok:
            eos_token_id = 0

            @classmethod
            def from_pretrained(cls, *a, **k):
                return cls()

            def apply_chat_template(self, messages, **kw):
                return "<|im_start|>" + messages[0]["content"]

            def __call__(self, text, **kw):
                return {"input_ids": np.array([[1, 2, 3, 4]])}

            def decode(self, tokens, **kw):
                return " answer: entitled under [Article 51]. "

        class _FakeModel:
            @classmethod
            def from_pretrained(cls, *a, **k):
                return cls()

            def generate(self, **kw):
                return np.array([[1, 2, 3, 4, 9, 9, 9]])

        fake_oi = types.ModuleType("optimum.intel")
        fake_oi.OVModelForCausalLM = _FakeModel
        monkeypatch.setitem(sys.modules, "optimum", types.ModuleType("optimum"))
        monkeypatch.setitem(sys.modules, "optimum.intel", fake_oi)
        monkeypatch.setattr("transformers.AutoTokenizer", _FakeTok, raising=False)

        llm = OpenVINOChatLLM(model_dir)
        out = llm.invoke("what is the gratuity")
        assert out == "answer: entitled under [Article 51]."

        runnable = llm.as_langchain()
        assert runnable.invoke("again") == "answer: entitled under [Article 51]."


# --------------------------------------------------------------------------- #
# Offline guarantees
# --------------------------------------------------------------------------- #
class TestBuildDefaultPipeline:
    def test_wires_store_embedder_and_llm(self, monkeypatch):
        from src import pipeline as pipeline_mod
        from src import vectorstore as vs_mod

        store = LegalVectorStore.build(DOCS, FixedEmbedder(TABLE, 3))
        monkeypatch.setattr(vs_mod, "load_default_embedder",
                            lambda: FixedEmbedder(TABLE, 3))
        monkeypatch.setattr(LegalVectorStore, "load",
                            classmethod(lambda cls, d, e: store))
        pipe = pipeline_mod.build_default_pipeline(llm=StubLLM())
        assert isinstance(pipe, HaqBotPipeline)
        assert pipe.answer("confident query").used_fallback is False


@pytest.mark.offline
class TestOffline:
    def test_full_answer_cycle_with_sockets_disabled(self, monkeypatch):
        import socket

        def _blocked(*_a, **_k):
            raise AssertionError("network access during pipeline.answer")

        monkeypatch.setattr(socket, "socket", _blocked)
        monkeypatch.setattr(socket, "create_connection", _blocked)

        pipe = make_pipeline(StubLLM())
        assert pipe.answer("confident query").used_fallback is False
        assert pipe.answer("weak query").used_fallback is True

    def test_import_pulls_no_network_or_ml_libs(self):
        code = (
            "import sys, src.pipeline; "
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
