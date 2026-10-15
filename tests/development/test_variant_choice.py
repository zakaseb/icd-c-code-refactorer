"""Target-ICD variant choice after analysis, and the local model default.

Detection is injected, so these tests do not call a model. The HTTP check
uses FastAPI's TestClient against a throwaway session directory.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "api"))
os.environ.setdefault("WORKSPACE_DIR", tempfile.mkdtemp(prefix="icd_variant_test_"))

import app as appmod  # noqa: E402

PASS = 0
FAIL = 0


def _check(label: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}: {detail}")


def test_normalize_keeps_only_multi_variant_peripherals():
    groups = appmod._normalize_variant_groups({
        "groups": [
            {"peripheral": "IMU", "variants": ["IMU15", "IMU15", " IMU20 "]},
            {"peripheral": "GPS", "variants": ["GPS-1"]},
            {"peripheral": "  ", "variants": ["A", "B"]},
            "nope",
        ]
    })
    _check("one_group", len(groups) == 1, repr(groups))
    _check(
        "deduped_names",
        groups and groups[0]["peripheral"] == "IMU"
        and groups[0]["variants"] == ["IMU15", "IMU20"],
        repr(groups),
    )
    _check(
        "empty_when_nothing_qualifies",
        appmod._normalize_variant_groups({"groups": []}) == [],
    )


def test_choice_pauses_then_continues_with_the_selected_variant():
    session = Path(tempfile.mkdtemp(prefix="icd_variant_session_"))
    calls = {"n": 0}

    def detect(_icd, _spec, _summary):
        calls["n"] += 1
        return [{"peripheral": "IMU", "variants": ["IMU15", "IMU20"]}]

    spec, event = appmod._resolve_variant_choice(
        session, "target icd", "apply the new message layout", "summary",
        detect_groups=detect,
    )
    _check("detector_called_once", calls["n"] == 1, str(calls["n"]))
    _check("paused", event is not None and event["type"] == "variant_choice_required")
    _check(
        "groups_in_event",
        event and event["groups"][0]["variants"] == ["IMU15", "IMU20"],
        repr(event),
    )
    status = appmod._read_status(session)
    _check(
        "awaiting_state",
        status.get("state") == "awaiting_variant_choice",
        repr(status.get("state")),
    )
    _check("spec_unchanged_until_choice", "SELECTED" not in spec, spec)

    spec2, event2 = appmod._resolve_variant_choice(
        session, "target icd", spec, "summary", detect_groups=detect,
    )
    _check("detector_not_repeated", calls["n"] == 1, str(calls["n"]))
    _check("still_paused", event2 is not None and "SELECTED" not in spec2)

    (session / "change_spec.txt").write_text(
        "apply the new message layout\n", encoding="utf-8",
    )
    try:
        appmod._save_variant_selection(session, {"IMU": "not-a-variant"})
        rejected = False
    except ValueError:
        rejected = True
    _check("rejects_unknown_variant", rejected)

    saved = appmod._save_variant_selection(session, {"IMU": "IMU20"})
    _check("saves_choice", saved == {"IMU": "IMU20"}, repr(saved))
    on_disk = (session / "change_spec.txt").read_text(encoding="utf-8")
    _check("spec_names_choice", "use IMU20" in on_disk, on_disk)
    _check("spec_excludes_other", "Do not use IMU15" in on_disk, on_disk)
    _check("spec_keeps_original", "apply the new message layout" in on_disk, on_disk)

    spec3, event3 = appmod._resolve_variant_choice(
        session, "target icd", on_disk, "summary", detect_groups=detect,
    )
    _check("continues_after_choice", event3 is None)
    _check("annotation_once", spec3.count(appmod.VARIANT_CHOICE_MARKER) == 1, spec3)
    _check(
        "selected_names",
        appmod._selected_variant_names(session) == {"IMU20"},
    )


def test_named_configurations_pause_even_if_the_model_says_none():
    """The IMU15 summary names two housings. A change spec that calls that
    one interface must not hide the choice."""
    session = Path(tempfile.mkdtemp(prefix="icd_variant_housings_"))
    summary = (
        "# Technical Summary: Interface Control Document for IMU15\n"
        "The IMU15 is a digital inertial measurement unit. "
        "It is available in two housing configurations: "
        "Standard Pod and Anti-Vibration (AV) Mounted Housing.\n"
        "Byte Format: 1 start bit, 8 data bits, 2 stop bits.\n"
    )
    spec = (
        "*   **Target (IMU15):** Single logical variation. "
        "Mechanical housing differences (Standard/AV) do not affect "
        "electrical/data interface.\n"
    )

    def detect(_icd, _change, _summary):
        return []

    groups = appmod._extract_variant_groups_from_text(summary)
    _check(
        "summary_names_housings",
        groups and groups[0]["peripheral"] == "IMU15"
        and groups[0]["variants"] == [
            "Standard Pod",
            "Anti-Vibration (AV) Mounted Housing",
        ],
        repr(groups),
    )
    _check(
        "stop_bits_are_not_variants",
        "2 stop bits" not in json.dumps(groups),
        repr(groups),
    )
    bullets = (
        "Interface Control Document for IMU-FI\n"
        "requirements for the following Inertial Measurement Unit (IMU):\n"
        "• IMU-FI-200C\n"
        "• IMU-FI-200T.\n"
    )
    bullet_groups = appmod._extract_variant_groups_from_text(bullets)
    _check(
        "bullet_part_numbers",
        bullet_groups and bullet_groups[0]["variants"] == ["IMU-FI-200C", "IMU-FI-200T"],
        repr(bullet_groups),
    )
    _, event = appmod._resolve_variant_choice(
        session, "target body without a list", spec, summary, detect_groups=detect,
    )
    _check("choice_event", event is not None and event["type"] == "variant_choice_required")
    _check(
        "choice_is_the_housings",
        event and event["groups"][0]["variants"] == [
            "Standard Pod",
            "Anti-Vibration (AV) Mounted Housing",
        ],
        repr(event),
    )


def test_same_variant_list_is_one_question():
    """IMU-FI-200C/T stated in the Target ICD must not be asked again under
    a second heading the model invents for the same two part numbers."""
    session = Path(tempfile.mkdtemp(prefix="icd_variant_once_"))
    target = (
        "requirements for the following Inertial Measurement Unit (IMU):\n"
        "• IMU-FI-200C\n"
        "• IMU-FI-200T.\n"
    )

    def detect(_icd, _spec, _summary):
        return [{
            "peripheral": "IMU-FI",
            "variants": ["IMU-FI-200T", "IMU-FI-200C"],
        }]

    _, event = appmod._resolve_variant_choice(
        session, target, "change spec", "", detect_groups=detect,
    )
    groups = event["groups"] if event else []
    _check("one_question", len(groups) == 1, repr(groups))
    _check(
        "keeps_the_text_heading",
        groups and groups[0]["peripheral"] == "Peripheral"
        and groups[0]["variants"] == ["IMU-FI-200C", "IMU-FI-200T"],
        repr(groups),
    )
    distinct = appmod._merge_variant_groups(
        [{"peripheral": "IMU15", "variants": ["Standard Pod", "AV Mount"]}],
        [{"peripheral": "GPS", "variants": ["GPS-1", "GPS-2"]}],
    )
    _check(
        "different_lists_stay_separate",
        [g["peripheral"] for g in distinct] == ["IMU15", "GPS"],
        repr(distinct),
    )


def test_single_variant_does_not_pause():
    session = Path(tempfile.mkdtemp(prefix="icd_variant_single_"))

    def detect(_icd, _spec, _summary):
        return []

    spec, event = appmod._resolve_variant_choice(
        session, "one device", "no variants here", "", detect_groups=detect,
    )
    _check("no_event", event is None)
    _check("spec_untouched", spec == "no variants here", spec)
    _check("groups_recorded", (session / appmod.VARIANT_GROUPS_FILE).is_file())
    _check(
        "resume_flag",
        appmod._is_pipeline_resume({"state": "awaiting_variant_choice"})
        and appmod._is_pipeline_resume({"state": "resuming"})
        and not appmod._is_pipeline_resume({"state": "processing"}),
    )


def test_variant_choice_endpoint():
    from fastapi.testclient import TestClient

    session_id = "variant-choice-http"
    session = appmod.SESSIONS_DIR / session_id
    session.mkdir(parents=True, exist_ok=True)
    (session / appmod.VARIANT_GROUPS_FILE).write_text(json.dumps({
        "groups": [{"peripheral": "IMU", "variants": ["IMU15", "IMU20"]}],
    }), encoding="utf-8")
    (session / "change_spec.txt").write_text("base spec\n", encoding="utf-8")
    appmod._write_status(session, {"state": "awaiting_variant_choice"})
    client = TestClient(appmod.app)

    missing = client.post("/api/variant-choice/does-not-exist", json={"selections": {}})
    _check("missing_session_404", missing.status_code == 404, str(missing.status_code))

    bad = client.post(
        f"/api/variant-choice/{session_id}",
        json={"selections": {"IMU": "IMU99"}},
    )
    _check("bad_choice_400", bad.status_code == 400, bad.text)

    ok = client.post(
        f"/api/variant-choice/{session_id}",
        json={"selections": {"IMU": "IMU15"}},
    )
    _check("choice_200", ok.status_code == 200, ok.text)
    body = ok.json()
    _check("choice_resuming", body.get("state") == "resuming", repr(body))
    status = appmod._read_status(session)
    _check(
        "status_resuming",
        status.get("state") == "resuming" and status.get("awaiting_variant_choice") is False,
        repr(status),
    )
    spec = (session / "change_spec.txt").read_text(encoding="utf-8")
    _check("endpoint_annotates_spec", "use IMU15" in spec and "base spec" in spec, spec)


class _FakeStreamResponse:
    def __init__(self):
        self.status_code = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_lines(self):
        payload = {
            "choices": [{
                "delta": {"content": "ok"},
                "finish_reason": "stop",
            }],
        }
        yield "data: " + json.dumps(payload)
        yield "data: [DONE]"


class _FakeClient:
    seen = []

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def stream(self, method, url, json=None, headers=None):
        _FakeClient.seen.append(json)
        return _FakeStreamResponse()


def test_process_stream_stops_before_transform():
    """A resumed analysis with two variants ends on the choice event."""
    from fastapi.testclient import TestClient

    session_id = "variant-choice-stream"
    session = appmod.SESSIONS_DIR / session_id
    code = session / "original_code"
    code.mkdir(parents=True, exist_ok=True)
    (code / "demo.c").write_text("int demo(void) { return 0; }\n", encoding="utf-8")
    (session / "source_icd.txt").write_text("source device\n", encoding="utf-8")
    (session / "target_icd.txt").write_text(
        "The peripheral has variants IMU15 and IMU20.\n", encoding="utf-8",
    )
    (session / "change_spec.txt").write_text("update the message\n", encoding="utf-8")
    (session / "target_summary.txt").write_text("two IMUs\n", encoding="utf-8")
    (session / "icd_analysis.txt").write_text("analysis\n", encoding="utf-8")
    (session / appmod.VARIANT_GROUPS_FILE).write_text(json.dumps({
        "groups": [{"peripheral": "IMU", "variants": ["IMU15", "IMU20"]}],
    }), encoding="utf-8")
    appmod._write_status(session, {
        "state": "resuming",
        "files": ["demo.c"],
        "source_icd": "source.pdf",
        "target_icd": "target.pdf",
        "pipeline_state": {
            "completed_stages": ["analysis"],
            "completed_files": [],
            "events": 1,
        },
    })
    original_ready = appmod._wait_for_llm_ready
    appmod._wait_for_llm_ready = lambda timeout_s=300: None
    try:
        client = TestClient(appmod.app)
        with client.stream("GET", f"/api/process/{session_id}") as response:
            body = "".join(response.iter_text())
    finally:
        appmod._wait_for_llm_ready = original_ready
    events = []
    for line in body.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    types = [event.get("type") for event in events]
    stages = [
        event.get("stage") for event in events if event.get("type") == "stage"
    ]
    _check(
        "stream_asks_for_choice",
        "variant_choice_required" in types,
        repr(types),
    )
    _check(
        "stream_stops_before_transform",
        "transform" not in stages,
        repr(stages),
    )
    choice = next(
        event for event in events if event.get("type") == "variant_choice_required"
    )
    _check(
        "stream_groups",
        choice["groups"][0]["variants"] == ["IMU15", "IMU20"],
        repr(choice),
    )
    status = appmod._read_status(session)
    _check(
        "stream_awaits",
        status.get("state") == "awaiting_variant_choice",
        repr(status.get("state")),
    )


def test_chat_request_disables_thinking_and_names_qwen38():
    _check(
        "default_model",
        appmod.HF_MODEL == "Qwen3.8-27B-UD-Q4_K_XL.gguf",
        appmod.HF_MODEL,
    )
    _check(
        "thinking_off",
        appmod.LLM_CHAT_TEMPLATE_KWARGS.get("enable_thinking") is False,
        repr(appmod.LLM_CHAT_TEMPLATE_KWARGS),
    )
    original = appmod.httpx.Client
    appmod.httpx.Client = _FakeClient
    _FakeClient.seen = []
    try:
        chunks = list(appmod._call_llm_stream("system", "user", max_tokens=32))
    finally:
        appmod.httpx.Client = original
    _check("stream_yields", chunks == ["ok"], repr(chunks))
    sent = _FakeClient.seen[0] if _FakeClient.seen else {}
    _check(
        "request_kwargs",
        sent.get("chat_template_kwargs") == {"enable_thinking": False},
        repr(sent.get("chat_template_kwargs")),
    )
    _check(
        "request_model",
        sent.get("model") == "Qwen3.8-27B-UD-Q4_K_XL",
        repr(sent.get("model")),
    )


def main() -> int:
    print("Running test_variant_choice")
    test_normalize_keeps_only_multi_variant_peripherals()
    test_choice_pauses_then_continues_with_the_selected_variant()
    test_named_configurations_pause_even_if_the_model_says_none()
    test_same_variant_list_is_one_question()
    test_single_variant_does_not_pause()
    test_variant_choice_endpoint()
    test_process_stream_stops_before_transform()
    test_chat_request_disables_thinking_and_names_qwen38()
    print(f"{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
