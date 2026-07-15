"""Execution context shared by every team in the agentic pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _default_app():
    """Lazy handle to the classic pipeline module (api/app.py).

    Teams reuse the proven stage helpers (context builders, completeness
    critics, compile/build machinery, RTOS/cross-toolchain detection)
    through this handle instead of re-implementing them, which keeps the
    agentic pipeline's outputs byte-compatible with the classic one.
    Tests substitute a stub namespace here.
    """
    import app as _app
    return _app


@dataclass
class PipelineContext:
    session_dir: Path
    gen_dir: Path
    code_dir: Path
    repo_dir: Path
    has_repo: bool
    source_icd: str
    target_icd: str
    llm: Any                      # object with .complete(system, user, **kw)
    app: Any = None               # classic-pipeline module (or test stub)
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.app is None:
            self.app = _default_app()

    @property
    def code_files(self) -> list[Path]:
        if not self.code_dir.exists():
            return []
        return sorted(p for p in self.code_dir.iterdir() if p.is_file())

    @property
    def uploaded_names(self) -> set[str]:
        return {p.name for p in self.code_files}
