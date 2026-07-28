"""Live deliverables during sandbox build: sync sandbox edits → generated_code."""

from __future__ import annotations

import ast
import io
import tempfile
import zipfile
from pathlib import Path


def _load_helpers():
    """Load sync helpers from api/app.py without importing the full FastAPI app."""
    src_path = Path(__file__).resolve().parents[2] / "api" / "app.py"
    src = src_path.read_text()
    tree = ast.parse(src)
    wanted = {
        "_sync_sandbox_deliverables",
        "_flush_sandbox_build_log",
        "_deliverables_file_list",
    }
    chunks = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            chunks.append(ast.get_source_segment(src, node))
    ns: dict = {"Path": Path}

    class _Log:
        def warning(self, *a, **k):
            pass

    ns["log"] = _Log()
    exec("\n\n".join(chunks), ns)
    return ns


def test_sync_copies_sandbox_edits_into_gen_dir(tmp_path: Path | None = None, helpers=None):
    helpers = helpers or _load_helpers()
    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())
    gen_dir = tmp_path / "generated_code"
    sandbox = tmp_path / "sandbox" / "src"
    gen_dir.mkdir(parents=True, exist_ok=True)
    sandbox.mkdir(parents=True, exist_ok=True)
    (gen_dir / "comms.c").write_text("OLD\n")
    sandbox_file = sandbox / "comms.c"
    sandbox_file.write_text("NEW_EDITED\n")
    replacement_map = {"comms.c": sandbox_file}

    synced = helpers["_sync_sandbox_deliverables"](replacement_map, gen_dir)
    assert synced == ["comms.c"]
    assert (gen_dir / "comms.c").read_text() == "NEW_EDITED\n"


def test_sync_skips_unchanged(tmp_path: Path | None = None, helpers=None):
    helpers = helpers or _load_helpers()
    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())
    gen_dir = tmp_path / "generated_code"
    sandbox = tmp_path / "sandbox"
    gen_dir.mkdir(parents=True, exist_ok=True)
    sandbox.mkdir(parents=True, exist_ok=True)
    (gen_dir / "a.h").write_text("SAME\n")
    (sandbox / "a.h").write_text("SAME\n")
    synced = helpers["_sync_sandbox_deliverables"](
        {"a.h": sandbox / "a.h"}, gen_dir,
    )
    assert synced == []


def test_download_zip_reflects_mid_build_edit(tmp_path: Path | None = None, helpers=None):
    """After sync, a download ZIP built like /api/download must contain edits."""
    helpers = helpers or _load_helpers()
    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())
    session = tmp_path / "session"
    gen_dir = session / "generated_code"
    sandbox = session / "sandbox"
    gen_dir.mkdir(parents=True)
    sandbox.mkdir()
    (gen_dir / "imu.c").write_text("version=1\n")
    (gen_dir / "verification_report.txt").write_text("ok\n")
    (session / "change_spec.txt").write_text("spec\n")
    (sandbox / "imu.c").write_text("version=2-mid-build\n")

    helpers["_sync_sandbox_deliverables"](
        {"imu.c": sandbox / "imu.c"}, gen_dir,
    )
    helpers["_flush_sandbox_build_log"](
        session, ["SANDBOX BUILD LOG", "attempt 3"],
    )

    assert (gen_dir / "imu.c").read_text() == "version=2-mid-build\n"
    assert "attempt 3" in (session / "sandbox_build_log.txt").read_text()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(gen_dir.iterdir()):
            if f.is_file() and f.name != "verification_report.txt":
                zf.write(f, f.name)
        zf.write(gen_dir / "verification_report.txt", "verification_report.txt")
        zf.write(session / "change_spec.txt", "change_spec.txt")
        zf.write(session / "sandbox_build_log.txt", "sandbox_build_log.txt")
    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        names = set(zf.namelist())
        assert "imu.c" in names
        assert zf.read("imu.c").decode() == "version=2-mid-build\n"
        assert "attempt 3" in zf.read("sandbox_build_log.txt").decode()
        assert "verification_report.txt" in names
        assert "change_spec.txt" in names


def test_deliverables_file_list_skips_objects(tmp_path: Path | None = None, helpers=None):
    helpers = helpers or _load_helpers()
    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())
    gen_dir = tmp_path / "generated_code"
    gen_dir.mkdir(parents=True, exist_ok=True)
    (gen_dir / "a.c").write_text("x\n")
    (gen_dir / "a.o").write_bytes(b"\x00\x01")
    (gen_dir / "compile_report.txt").write_text("r\n")
    names = helpers["_deliverables_file_list"](gen_dir)
    assert names == ["a.c", "compile_report.txt"]


if __name__ == "__main__":
    h = _load_helpers()
    test_sync_copies_sandbox_edits_into_gen_dir(helpers=h)
    test_sync_skips_unchanged(helpers=h)
    test_download_zip_reflects_mid_build_edit(helpers=h)
    test_deliverables_file_list_skips_objects(helpers=h)
    print("ALL TESTS PASSED")
