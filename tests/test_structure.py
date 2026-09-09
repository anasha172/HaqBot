"""Phase 1 — repository structure & tooling contract.

Verifies the folder layout, config files, and pinned dependencies mandated by
PRD §5 (Phase 1) and §6 are present and well-formed.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from src import config

ROOT = config.BASE_DIR


@pytest.mark.parametrize(
    "relpath",
    [
        "data/raw",
        "data/processed",
        "models",
        "src",
        "tests",
        ".streamlit",
    ],
)
def test_required_directories_exist(relpath):
    assert (ROOT / relpath).is_dir(), f"missing directory: {relpath}"


@pytest.mark.parametrize(
    "relpath",
    [
        "requirements.txt",
        "README.md",
        "PRD.md",
        "pytest.ini",
        ".gitignore",
        ".streamlit/config.toml",
        "src/__init__.py",
        "src/config.py",
    ],
)
def test_required_files_exist(relpath):
    assert (ROOT / relpath).is_file(), f"missing file: {relpath}"


class TestGitkeeps:
    @pytest.mark.parametrize(
        "relpath",
        ["data/raw/.gitkeep", "data/processed/.gitkeep", "models/.gitkeep"],
    )
    def test_gitkeep_present(self, relpath):
        assert (ROOT / relpath).is_file()


class TestGitignore:
    @pytest.fixture
    def ignore_text(self):
        return (ROOT / ".gitignore").read_text(encoding="utf-8")

    @pytest.mark.parametrize(
        "pattern",
        ["data/local_user.db", "models/*", "data/processed/*", "__pycache__/",
         ".venv/"],
    )
    def test_sensitive_paths_ignored(self, ignore_text, pattern):
        assert pattern in ignore_text


class TestRequirements:
    @pytest.fixture
    def requirements(self):
        lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]

    @pytest.mark.parametrize(
        "package",
        ["openvino", "optimum-intel", "langchain", "faiss-cpu", "streamlit",
         "PyMuPDF", "pytest"],
    )
    def test_core_packages_pinned(self, requirements, package):
        matches = [r for r in requirements if r.lower().startswith(package.lower())]
        assert matches, f"{package} not in requirements.txt"
        assert "==" in matches[0], f"{package} is not pinned with =="

    def test_no_unpinned_requirements(self, requirements):
        for req in requirements:
            assert "==" in req, f"unpinned requirement: {req}"


class TestStreamlitConfigParses:
    def test_toml_is_valid(self):
        with open(ROOT / ".streamlit" / "config.toml", "rb") as fh:
            data = tomllib.load(fh)
        assert "theme" in data and "browser" in data and "server" in data


class TestPrdAlignment:
    """Spot-checks that the code mirrors the numbers written in the PRD."""

    @pytest.fixture
    def prd_text(self):
        return (ROOT / "PRD.md").read_text(encoding="utf-8")

    def test_threshold_matches_prd(self, prd_text):
        assert "0.65" in prd_text
        assert config.SIMILARITY_THRESHOLD == 0.65

    def test_chunk_params_match_prd(self, prd_text):
        assert "450" in prd_text and "50" in prd_text
        assert (config.CHUNK_SIZE, config.CHUNK_OVERLAP) == (450, 50)

    def test_helpline_matches_prd(self, prd_text):
        assert "80084" in prd_text
        assert config.MOHRE_HELPLINE == "80084"
