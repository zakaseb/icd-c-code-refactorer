#!/usr/bin/env python3
"""Download the configured GGUF model into $HOME/models if missing.

Used by deployments/docker/entrypoint.sh on container start. Requires
huggingface_hub (installed into /opt/venv in the Docker image).
"""

from __future__ import annotations

import glob
import os
import sys


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        print(f"ERROR: required environment variable {name} is unset or empty.", file=sys.stderr)
        sys.exit(1)
    return value


def _find_model(local_repo_path: str, model_name: str) -> str | None:
    """Return an existing path to *model_name* under *local_repo_path*, if any."""
    exact = os.path.join(local_repo_path, model_name)
    if os.path.isfile(exact):
        return exact

    # Non-recursive then recursive: HF layouts sometimes nest shards/files.
    patterns = (
        os.path.join(local_repo_path, f"*{model_name}"),
        os.path.join(local_repo_path, "**", model_name),
        os.path.join(local_repo_path, "**", f"*{model_name}"),
    )
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern, recursive=True))
    files = sorted({m for m in matches if os.path.isfile(m)})
    return files[0] if files else None


def main() -> None:
    model_name = _require_env("HF_MODEL")
    repo_id = _require_env("HF_REPO_ID")
    user_home = _require_env("HOME")

    local_repo_path = os.path.join(user_home, "models", repo_id)
    os.makedirs(local_repo_path, exist_ok=True)

    found = _find_model(local_repo_path, model_name)
    if found:
        print(f"\nFound model: {found}\n")
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        print(
            "ERROR: huggingface_hub is not importable. "
            "Activate /opt/venv (Docker) or install huggingface_hub.",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        sys.exit(1)

    allow_patterns = [model_name, f"*{model_name}"]
    print(f"Downloading {repo_id} pattern(s) {allow_patterns} → {local_repo_path}")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_repo_path,
            allow_patterns=allow_patterns,
        )
    except Exception as exc:  # noqa: BLE001 — surface HF errors clearly at boot
        print(f"ERROR: model download failed: {exc}", file=sys.stderr)
        sys.exit(1)

    found = _find_model(local_repo_path, model_name)
    if not found:
        print(
            f"ERROR: download finished but {model_name!r} was not found under {local_repo_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nDownloaded model: {found}\n")


if __name__ == "__main__":
    main()
