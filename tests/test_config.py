"""Phase 1 — validation of src/config.py and the Streamlit config.

These tests are pure and offline: they import a side-effect-free module and
parse a local TOML file. No network, no model, no database.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

from src import config


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
class TestPaths:
    def test_base_dir_is_repo_root(self):
        assert (config.BASE_DIR / "PRD.md").is_file()
        assert (config.BASE_DIR / "requirements.txt").is_file()

    @pytest.mark.parametrize(
        "path",
        [
            config.DATA_DIR,
            config.RAW_DIR,
            config.PROCESSED_DIR,
            config.MODELS_DIR,
            config.DB_PATH,
            config.FAISS_INDEX_DIR,
            config.FAISS_INDEX_PATH,
            config.CHUNKS_META_PATH,
            config.EMBEDDING_MODEL_DIR,
            config.LLM_MODEL_DIR,
            config.STREAMLIT_CONFIG_PATH,
        ],
    )
    def test_paths_are_path_objects_under_base_dir(self, path):
        assert isinstance(path, Path)
        assert config.BASE_DIR in path.parents

    def test_db_lives_in_data_dir(self):
        assert config.DB_PATH.parent == config.DATA_DIR
        assert config.DB_PATH.name == "local_user.db"

    def test_all_directories_tuple_is_consistent(self):
        assert config.RAW_DIR in config.ALL_DIRECTORIES
        assert config.PROCESSED_DIR in config.ALL_DIRECTORIES
        assert config.MODELS_DIR in config.ALL_DIRECTORIES
        for directory in config.ALL_DIRECTORIES:
            assert isinstance(directory, Path)


# --------------------------------------------------------------------------- #
# Import hygiene — config must be free of side effects
# --------------------------------------------------------------------------- #
class TestImportHygiene:
    def test_fresh_import_touches_no_disk(self):
        """Importing config in a clean subprocess must create nothing on disk."""
        import subprocess

        code = (
            "import sys; "
            "before = {p for p in __import__('pathlib').Path('.').rglob('*')}; "
            "import src.config; "
            "after = {p for p in __import__('pathlib').Path('.').rglob('*')}; "
            "print(sorted(str(p) for p in after - before))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=config.BASE_DIR,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "[]", result.stdout

    def test_config_import_pulls_no_ml_or_network_libs(self):
        """A subprocess that imports only config must not load heavy libs."""
        import subprocess

        code = (
            "import sys; import src.config; "
            "bad = {'openvino','torch','transformers','streamlit','faiss',"
            "'requests','httpx','sentence_transformers'} & set(sys.modules); "
            "print(sorted(bad))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=config.BASE_DIR,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "[]", result.stdout


# --------------------------------------------------------------------------- #
# Retrieval + guardrail thresholds
# --------------------------------------------------------------------------- #
class TestThresholds:
    def test_similarity_threshold_is_prd_value(self):
        assert config.SIMILARITY_THRESHOLD == 0.65
        assert isinstance(config.SIMILARITY_THRESHOLD, float)
        assert 0.0 < config.SIMILARITY_THRESHOLD < 1.0

    def test_chunk_sizes_match_prd(self):
        assert config.CHUNK_SIZE == 450
        assert config.CHUNK_OVERLAP == 50
        assert config.CHUNK_OVERLAP < config.CHUNK_SIZE

    def test_retrieval_top_k_is_positive(self):
        assert isinstance(config.RETRIEVAL_TOP_K, int)
        assert config.RETRIEVAL_TOP_K >= 1

    def test_generation_is_deterministic_by_default(self):
        assert config.TEMPERATURE == 0.0
        assert config.MAX_NEW_TOKENS > 0


# --------------------------------------------------------------------------- #
# Model configuration
# --------------------------------------------------------------------------- #
class TestModels:
    def test_int8_weight_format(self):
        assert config.LLM_WEIGHT_FORMAT == "int8"

    def test_openvino_targets_cpu(self):
        assert config.OPENVINO_DEVICE == "CPU"

    def test_model_ids_are_non_empty(self):
        assert config.EMBEDDING_MODEL_ID
        assert "e5" in config.EMBEDDING_MODEL_ID.lower()
        assert "qwen" in config.LLM_MODEL_ID.lower()

    def test_e5_prefixes_present(self):
        assert config.E5_QUERY_PREFIX.strip() == "query:"
        assert config.E5_PASSAGE_PREFIX.strip() == "passage:"


# --------------------------------------------------------------------------- #
# Localisation
# --------------------------------------------------------------------------- #
class TestLocalisation:
    def test_five_supported_languages(self):
        assert set(config.SUPPORTED_LANGUAGES) == {"en", "hi", "ur", "ml", "ar"}

    def test_default_language_is_supported(self):
        assert config.DEFAULT_LANGUAGE in config.SUPPORTED_LANGUAGES

    def test_rtl_languages_are_a_subset(self):
        assert config.RTL_LANGUAGES <= set(config.SUPPORTED_LANGUAGES)
        assert config.RTL_LANGUAGES == {"ur", "ar"}

    def test_is_rtl_helper(self):
        assert config.is_rtl("ar") is True
        assert config.is_rtl("ur") is True
        assert config.is_rtl("en") is False
        assert config.is_rtl("hi") is False

    def test_language_name_helper(self):
        assert config.language_name("ml") == "മലയാളം"
        assert config.language_name("zz") == config.SUPPORTED_LANGUAGES["en"]

    def test_font_scale_options(self):
        assert config.DEFAULT_FONT_SCALE in config.FONT_SCALE_OPTIONS
        assert config.FONT_SCALE_OPTIONS[config.DEFAULT_FONT_SCALE] == 1.0
        assert all(v > 0 for v in config.FONT_SCALE_OPTIONS.values())


# --------------------------------------------------------------------------- #
# Auth constants
# --------------------------------------------------------------------------- #
class TestAuthConstants:
    def test_pin_length_is_four(self):
        assert config.PIN_LENGTH == 4

    def test_hash_algorithm_declared(self):
        assert config.PIN_HASH_ALGORITHM == "sha256"
        assert config.PBKDF2_ITERATIONS >= 100_000

    def test_guest_username(self):
        assert config.GUEST_USERNAME == "guest"


# --------------------------------------------------------------------------- #
# Guardrail messaging + system prompt
# --------------------------------------------------------------------------- #
class TestGuardrailMessaging:
    def test_helpline_and_website(self):
        assert config.MOHRE_HELPLINE == "80084"
        assert config.MOHRE_WEBSITE == "www.mohre.gov.ae"

    def test_fallback_message_contents(self):
        msg = config.FALLBACK_MESSAGE
        assert "80084" in msg
        assert "www.mohre.gov.ae" in msg
        assert "offline database" in msg

    def test_system_prompt_has_runtime_placeholders(self):
        assert "{context}" in config.SYSTEM_PROMPT
        assert "{query}" in config.SYSTEM_PROMPT

    def test_system_prompt_enforces_citations(self):
        assert "[Article X, Clause Y]" in config.SYSTEM_PROMPT

    def test_system_prompt_embeds_exact_refusal(self):
        # The exact refusal string from the PRD must be reproduced verbatim.
        assert (
            "I do not have sufficient legal documentation in my offline database"
            in config.SYSTEM_PROMPT
        )

    def test_system_prompt_formats_without_keyerror(self):
        rendered = config.SYSTEM_PROMPT.format(context="CTX", query="Q")
        assert "CTX" in rendered
        assert "Q" in rendered

    def test_disclaimer_mentions_law_and_mohre(self):
        assert "Decree-Law No. 33" in config.LEGAL_DISCLAIMER
        assert "80084" in config.LEGAL_DISCLAIMER


# --------------------------------------------------------------------------- #
# Air-gap enforcement
# --------------------------------------------------------------------------- #
class TestAirGap:
    def test_offline_only_flag(self):
        assert config.OFFLINE_ONLY is True

    @pytest.mark.parametrize(
        "key",
        ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"],
    )
    def test_offline_env_disables_hub(self, key):
        assert config.OFFLINE_ENV[key] == "1"


# --------------------------------------------------------------------------- #
# ensure_directories()
# --------------------------------------------------------------------------- #
class TestEnsureDirectories:
    def test_creates_directories(self, tmp_path, monkeypatch):
        fake_dirs = tuple(tmp_path / name for name in ("raw", "processed", "idx"))
        monkeypatch.setattr(config, "ALL_DIRECTORIES", fake_dirs)
        for d in fake_dirs:
            assert not d.exists()
        config.ensure_directories()
        for d in fake_dirs:
            assert d.is_dir()

    def test_is_idempotent(self, tmp_path, monkeypatch):
        fake_dirs = (tmp_path / "a", tmp_path / "b")
        monkeypatch.setattr(config, "ALL_DIRECTORIES", fake_dirs)
        config.ensure_directories()
        config.ensure_directories()  # must not raise
        for d in fake_dirs:
            assert d.is_dir()


# --------------------------------------------------------------------------- #
# .streamlit/config.toml
# --------------------------------------------------------------------------- #
class TestStreamlitConfig:
    @pytest.fixture
    def toml_data(self):
        with open(config.STREAMLIT_CONFIG_PATH, "rb") as fh:
            return tomllib.load(fh)

    def test_config_file_exists(self):
        assert config.STREAMLIT_CONFIG_PATH.is_file()

    def test_telemetry_disabled(self, toml_data):
        assert toml_data["browser"]["gatherUsageStats"] is False

    def test_no_raw_tracebacks(self, toml_data):
        assert toml_data["client"]["showErrorDetails"] in ("none", False)

    def test_server_is_headless_and_local(self, toml_data):
        assert toml_data["server"]["headless"] is True
        assert toml_data["server"]["enableXsrfProtection"] is True

    def test_theme_uses_rose_accent(self, toml_data):
        theme = toml_data["theme"]
        assert theme["base"] == "light"
        assert theme["primaryColor"].lower() == "#e11d48"
        assert theme["backgroundColor"].lower() == "#ffffff"
        assert theme["secondaryBackgroundColor"].lower() == "#f5f6f8"
        assert theme["textColor"].lower() == "#14161b"

    def test_theme_tokens_match_config_module(self, toml_data):
        assert toml_data["theme"]["primaryColor"].lower() == config.THEME["accent"]
        assert toml_data["theme"]["backgroundColor"].lower() == config.THEME["card"]
