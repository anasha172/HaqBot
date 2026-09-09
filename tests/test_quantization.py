"""Phase 4 — optimum-cli OpenVINO export orchestration.

No model download and no OpenVINO install: the export command is verified as
data, and ``subprocess.run`` is faked so ``run_export`` can be exercised end to
end (skip-if-present, success, failure, missing-binary) without touching the
network.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys

import pytest

from src import config
from src import quantization as q
from src.quantization import (
    EMBEDDING_SPEC,
    LLM_SPEC,
    QuantizationError,
    build_export_command,
    export_all,
    is_exported,
    run_export,
    verify_export,
)


def _fake_completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(
        args=["optimum-cli"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _write_ir(directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    (directory / "openvino_model.bin").write_bytes(b"\x00\x01\x02\x03")


# --------------------------------------------------------------------------- #
# build_export_command
# --------------------------------------------------------------------------- #
class TestBuildCommand:
    def test_embedding_command_shape(self):
        cmd = build_export_command(EMBEDDING_SPEC)
        assert cmd[:3] == [config.OPTIMUM_CLI, "export", "openvino"]
        assert "--model" in cmd and config.EMBEDDING_MODEL_ID in cmd
        assert cmd[cmd.index("--task") + 1] == config.EMBEDDING_EXPORT_TASK
        assert cmd[cmd.index("--weight-format") + 1] == "int8"
        assert cmd[-1] == str(config.EMBEDDING_MODEL_DIR)

    def test_llm_command_has_int8_ratio(self):
        cmd = build_export_command(LLM_SPEC)
        assert cmd[cmd.index("--weight-format") + 1] == "int8"
        assert cmd[cmd.index("--ratio") + 1] == config.LLM_INT8_RATIO
        assert cmd[cmd.index("--task") + 1] == "text-generation-with-past"
        assert cmd[-1] == str(config.LLM_MODEL_DIR)

    def test_optimum_cli_override(self):
        cmd = build_export_command(EMBEDDING_SPEC, optimum_cli="/opt/x/optimum-cli")
        assert cmd[0] == "/opt/x/optimum-cli"


# --------------------------------------------------------------------------- #
# is_exported / verify_export
# --------------------------------------------------------------------------- #
class TestExportState:
    def test_is_exported_false_when_absent(self, tmp_path):
        assert is_exported(tmp_path) is False

    def test_is_exported_true_when_files_present(self, tmp_path):
        _write_ir(tmp_path)
        assert is_exported(tmp_path) is True

    def test_is_exported_false_when_bin_empty(self, tmp_path):
        _write_ir(tmp_path)
        (tmp_path / "openvino_model.bin").write_bytes(b"")
        assert is_exported(tmp_path) is False

    def test_verify_export_raises_on_missing(self, tmp_path):
        (tmp_path / "openvino_model.xml").write_text("x", encoding="utf-8")
        with pytest.raises(QuantizationError):
            verify_export(tmp_path)

    def test_verify_export_raises_on_empty_bin(self, tmp_path):
        _write_ir(tmp_path)
        (tmp_path / "openvino_model.bin").write_bytes(b"")
        with pytest.raises(QuantizationError):
            verify_export(tmp_path)

    def test_verify_export_ok(self, tmp_path):
        _write_ir(tmp_path)
        assert verify_export(tmp_path) == tmp_path


# --------------------------------------------------------------------------- #
# run_export
# --------------------------------------------------------------------------- #
class TestRunExport:
    def test_skips_when_already_exported(self, tmp_path, monkeypatch):
        _write_ir(tmp_path)
        spec = dataclasses.replace(EMBEDDING_SPEC, output_dir=tmp_path)

        def _boom(*_a, **_k):
            raise AssertionError("subprocess should not run")

        monkeypatch.setattr(subprocess, "run", _boom)
        assert run_export(spec) == tmp_path

    def test_success_invokes_cli_and_verifies(self, tmp_path, monkeypatch):
        spec = dataclasses.replace(EMBEDDING_SPEC, output_dir=tmp_path / "out")
        seen = {}

        def _fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            _write_ir(tmp_path / "out")
            return _fake_completed(0, stdout="done")

        monkeypatch.setattr(subprocess, "run", _fake_run)
        out = run_export(spec, optimum_cli="optimum-cli")
        assert out == tmp_path / "out"
        assert seen["cmd"][:3] == ["optimum-cli", "export", "openvino"]

    def test_force_reexports_even_if_present(self, tmp_path, monkeypatch):
        _write_ir(tmp_path)
        spec = dataclasses.replace(EMBEDDING_SPEC, output_dir=tmp_path)
        calls = {"n": 0}

        def _fake_run(cmd, **kwargs):
            calls["n"] += 1
            _write_ir(tmp_path)
            return _fake_completed(0)

        monkeypatch.setattr(subprocess, "run", _fake_run)
        run_export(spec, force=True, optimum_cli="optimum-cli")
        assert calls["n"] == 1

    def test_nonzero_exit_raises_with_stderr_tail(self, tmp_path, monkeypatch):
        spec = dataclasses.replace(EMBEDDING_SPEC, output_dir=tmp_path / "out")
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **k: _fake_completed(1, stderr="line1\nBOOM: bad model"),
        )
        with pytest.raises(QuantizationError) as exc:
            run_export(spec, optimum_cli="optimum-cli")
        assert "BOOM: bad model" in str(exc.value)

    def test_missing_binary_raises_clean_error(self, tmp_path, monkeypatch):
        spec = dataclasses.replace(EMBEDDING_SPEC, output_dir=tmp_path / "out")

        def _no_binary(*_a, **_k):
            raise FileNotFoundError("optimum-cli")

        monkeypatch.setattr(subprocess, "run", _no_binary)
        with pytest.raises(QuantizationError):
            run_export(spec, optimum_cli="optimum-cli")

    def test_timeout_raises_clean_error(self, tmp_path, monkeypatch):
        spec = dataclasses.replace(EMBEDDING_SPEC, output_dir=tmp_path / "out")

        def _timeout(*_a, **_k):
            raise subprocess.TimeoutExpired(cmd="optimum-cli", timeout=1)

        monkeypatch.setattr(subprocess, "run", _timeout)
        with pytest.raises(QuantizationError):
            run_export(spec, optimum_cli="optimum-cli", timeout=1)

    def test_unavailable_optimum_raises_actionable_message(self, tmp_path, monkeypatch):
        spec = dataclasses.replace(EMBEDDING_SPEC, output_dir=tmp_path / "out")
        monkeypatch.setattr(q, "optimum_cli_available", lambda: False)
        with pytest.raises(QuantizationError) as exc:
            run_export(spec)  # no optimum_cli override
        assert "optimum-cli is not available" in str(exc.value)

    def test_export_all_runs_both_specs(self, tmp_path, monkeypatch):
        done = []
        monkeypatch.setattr(q, "run_export", lambda spec, **k: done.append(spec.name) or tmp_path)
        result = export_all(force=True)
        assert set(result) == {"embedding", "llm"}
        assert set(done) == {"embedding", "llm"}

    def test_helper_wrappers_delegate_to_correct_spec(self, tmp_path, monkeypatch):
        seen = []
        monkeypatch.setattr(q, "run_export", lambda spec, **k: seen.append(spec) or tmp_path)
        q.export_embedding_model()
        q.export_llm()
        assert [s.name for s in seen] == ["embedding", "llm"]

    def test_optimum_cli_available_returns_bool(self):
        assert isinstance(q.optimum_cli_available(), bool)


# --------------------------------------------------------------------------- #
# Offline import
# --------------------------------------------------------------------------- #
@pytest.mark.offline
class TestOffline:
    def test_import_pulls_no_network_libs(self):
        code = (
            "import sys, src.quantization; "
            "bad = {'requests','httpx','aiohttp','urllib.request','openvino',"
            "'optimum','torch','transformers'} & set(sys.modules); "
            "print(sorted(bad))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=config.BASE_DIR, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "[]", result.stdout
