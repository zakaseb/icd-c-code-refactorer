#!/usr/bin/env python3
"""Offline tests for src/utils/hf_download.py helpers."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HF_PATH = ROOT / "src" / "utils" / "hf_download.py"


def _load_mod():
    spec = importlib.util.spec_from_file_location("hf_download", HF_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class HfDownloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _load_mod()

    def test_find_exact_and_nested(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            nested = repo / "subdir"
            nested.mkdir()
            target = nested / "model.gguf"
            target.write_bytes(b"gguf")
            found = self.mod._find_model(str(repo), "model.gguf")
            self.assertEqual(Path(found), target)

            flat = repo / "other.gguf"
            flat.write_bytes(b"gguf")
            found_flat = self.mod._find_model(str(repo), "other.gguf")
            self.assertEqual(Path(found_flat), flat)

    def test_require_env(self) -> None:
        os.environ["HF_TEST_VAR"] = "yes"
        self.assertEqual(self.mod._require_env("HF_TEST_VAR"), "yes")
        del os.environ["HF_TEST_VAR"]
        with self.assertRaises(SystemExit):
            self.mod._require_env("HF_TEST_VAR")

    def test_main_reuses_existing_without_hub(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            repo_id = "org/model-gguf"
            model = "weights.gguf"
            dest = home / "models" / repo_id
            dest.mkdir(parents=True)
            (dest / model).write_bytes(b"gguf")
            env = {
                "HOME": str(home),
                "HF_REPO_ID": repo_id,
                "HF_MODEL": model,
            }
            old = {k: os.environ.get(k) for k in env}
            try:
                os.environ.update(env)
                self.mod.main()
            finally:
                for k, v in old.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v


if __name__ == "__main__":
    unittest.main()
