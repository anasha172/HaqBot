"""HaqBot — export Hugging Face models to INT8 OpenVINO IR (build-time step).

This is the ONE step that needs the network, and it is run once on a build
machine — never by end users, never at runtime:

    pip install -r requirements.txt          # installs openvino / optimum-intel
    python -m src.quantization all           # downloads + exports both models

Outputs:
    models/e5-small-ov/            multilingual-e5-small, INT8 OpenVINO IR
    models/qwen2.5-1.5b-ov-int8/   Qwen2.5-1.5B-Instruct, INT8 weight-compressed

At runtime HaqBot loads only these local folders (see src/vectorstore.py and,
from Phase 5, src/pipeline.py). Every failure raises :class:`QuantizationError`
with a plain message — no raw traceback.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from src import config

__all__ = [
    "QuantizationError",
    "ExportSpec",
    "EMBEDDING_SPEC",
    "LLM_SPEC",
    "build_export_command",
    "is_exported",
    "verify_export",
    "run_export",
    "export_embedding_model",
    "export_llm",
    "export_all",
    "optimum_cli_available",
]


class QuantizationError(Exception):
    """Any failure exporting or verifying an OpenVINO IR model."""


@dataclass(frozen=True)
class ExportSpec:
    """A single ``optimum-cli export openvino`` invocation."""

    name: str
    model_id: str
    task: str
    output_dir: Path
    weight_format: str
    extra_args: tuple[str, ...] = field(default_factory=tuple)


EMBEDDING_SPEC = ExportSpec(
    name="embedding",
    model_id=config.EMBEDDING_MODEL_ID,
    task=config.EMBEDDING_EXPORT_TASK,
    output_dir=config.EMBEDDING_MODEL_DIR,
    weight_format=config.EMBEDDING_WEIGHT_FORMAT,
    extra_args=("--library", "sentence_transformers"),
)

LLM_SPEC = ExportSpec(
    name="llm",
    model_id=config.LLM_MODEL_ID,
    task=config.LLM_EXPORT_TASK,
    output_dir=config.LLM_MODEL_DIR,
    weight_format=config.LLM_WEIGHT_FORMAT,
    extra_args=("--ratio", config.LLM_INT8_RATIO),
)

_SPECS: dict[str, ExportSpec] = {
    "embedding": EMBEDDING_SPEC,
    "llm": LLM_SPEC,
}


# =========================================================================== #
# Pure helpers
# =========================================================================== #
def build_export_command(
    spec: ExportSpec, *, optimum_cli: str | None = None
) -> list[str]:
    """Construct the ``optimum-cli export openvino`` argv for a spec."""
    return [
        optimum_cli or config.OPTIMUM_CLI,
        "export",
        "openvino",
        "--model",
        spec.model_id,
        "--task",
        spec.task,
        "--weight-format",
        spec.weight_format,
        *spec.extra_args,
        str(spec.output_dir),
    ]


def is_exported(directory: str | Path) -> bool:
    """True if every required IR file exists and is non-empty."""
    base = Path(directory)
    return all(
        (base / name).is_file() and (base / name).stat().st_size > 0
        for name in config.OV_MODEL_FILES
    )


def verify_export(directory: str | Path) -> Path:
    """Raise :class:`QuantizationError` unless the export looks complete."""
    base = Path(directory)
    missing = [n for n in config.OV_MODEL_FILES if not (base / n).is_file()]
    if missing:
        raise QuantizationError(
            f"Export in {base} is incomplete — missing {', '.join(missing)}."
        )
    empty = [
        n for n in config.OV_MODEL_FILES if (base / n).stat().st_size == 0
    ]
    if empty:
        raise QuantizationError(
            f"Export in {base} produced empty file(s): {', '.join(empty)}."
        )
    return base


def optimum_cli_available() -> bool:
    """True only if optimum is importable AND the CLI is on PATH."""
    return (
        importlib.util.find_spec("optimum") is not None
        and shutil.which(config.OPTIMUM_CLI) is not None
    )


# =========================================================================== #
# Export runners
# =========================================================================== #
def run_export(
    spec: ExportSpec,
    *,
    force: bool = False,
    optimum_cli: str | None = None,
    timeout: float | None = None,
) -> Path:
    """Export one model. Skips work if already exported (unless ``force``)."""
    out_dir = Path(spec.output_dir)
    if is_exported(out_dir) and not force:
        return out_dir

    if optimum_cli is None and not optimum_cli_available():
        raise QuantizationError(
            "optimum-cli is not available. Install the build dependencies "
            "(`pip install -r requirements.txt`) on a machine with network "
            "access, then re-run `python -m src.quantization`."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    command = build_export_command(spec, optimum_cli=optimum_cli)
    try:
        proc = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout
        )
    except FileNotFoundError as exc:
        raise QuantizationError(
            f"Could not launch '{command[0]}': {exc}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise QuantizationError(
            f"{spec.name} export timed out after {timeout}s."
        ) from exc

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()[-15:]
        raise QuantizationError(
            f"{spec.name} export failed (exit {proc.returncode}):\n"
            + "\n".join(detail)
        )
    return verify_export(out_dir)


def export_embedding_model(**kwargs) -> Path:
    return run_export(EMBEDDING_SPEC, **kwargs)


def export_llm(**kwargs) -> Path:
    return run_export(LLM_SPEC, **kwargs)


def export_all(*, force: bool = False, **kwargs) -> dict[str, Path]:
    return {
        name: run_export(spec, force=force, **kwargs)
        for name, spec in _SPECS.items()
    }


# =========================================================================== #
# CLI: python -m src.quantization [all|embedding|llm] [--force] [--check]
# =========================================================================== #
def _main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI
    import argparse

    parser = argparse.ArgumentParser(description="Export models to INT8 OpenVINO IR.")
    parser.add_argument(
        "target", nargs="?", default="all", choices=["all", "embedding", "llm"]
    )
    parser.add_argument("--force", action="store_true", help="re-export even if present")
    parser.add_argument(
        "--check", action="store_true", help="only report export status"
    )
    args = parser.parse_args(argv)

    specs = list(_SPECS.values()) if args.target == "all" else [_SPECS[args.target]]

    if args.check:
        for spec in specs:
            state = "ready" if is_exported(spec.output_dir) else "MISSING"
            print(f"  {state:8s} {spec.name:10s} -> {spec.output_dir}")
        return 0

    try:
        for spec in specs:
            print(f"exporting {spec.name} ({spec.model_id}) ...")
            path = run_export(spec, force=args.force)
            print(f"  ok -> {path}")
    except QuantizationError as exc:
        print(f"quantization failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
