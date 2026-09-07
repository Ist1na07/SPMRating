"""Guard test: the engine's embedded parameter set must match the release file.

`interface/python/spm_engine.py` carries a self-contained copy of the release
parameters (`_PARAMS_JSON`) so the interface works from a single file; the
authoritative copy lives at `params/spm-rating-1.0.0.json`.  Nothing else
keeps the two in sync, so drift would silently change the interface's star
ratings.  This test pins them together — run it (or the full pytest suite)
before any release.

Run:  python -m pytest tests/test_params_sync.py -q
  or  python tests/test_params_sync.py
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_ENGINE = os.path.join(_ROOT, "interface", "python", "spm_engine.py")
_PARAMS_FILE = os.path.join(_ROOT, "params", "spm-rating-1.0.0.json")


def _engine_params() -> dict:
    """Import the bundled engine and return its embedded parameter dict."""
    sys.path.insert(0, os.path.dirname(_ENGINE))
    try:
        import spm_engine  # noqa: PLC0415
        return dict(spm_engine._PARAMS_JSON["params"])
    finally:
        sys.path.pop(0)


def _file_params() -> dict:
    with open(_PARAMS_FILE, "r", encoding="utf-8") as fh:
        return dict(json.load(fh)["params"])


def test_param_key_sets_match() -> None:
    engine, release = _engine_params(), _file_params()
    assert sorted(engine) == sorted(release), (
        "embedded and release parameter files disagree on the key set; "
        f"embedded-only={sorted(set(engine) - set(release))} "
        f"release-only={sorted(set(release) - set(engine))}"
    )


def test_param_values_match_exactly() -> None:
    engine, release = _engine_params(), _file_params()
    drift = {k: (release[k], engine[k])
             for k in release if k in engine and release[k] != engine[k]}
    assert not drift, (
        "embedded engine parameters drifted from the release file: "
        + ", ".join(f"{k}: file={f!r} engine={e!r}" for k, (f, e) in drift.items())
    )


def test_param_counts_match() -> None:
    engine, release = _engine_params(), _file_params()
    assert len(engine) == len(release) == 248, (
        f"expected 248 parameters, got engine={len(engine)} file={len(release)}"
    )


def test_note_matches_release_scores() -> None:
    """The embedded note must quote the release file's benchmark scores."""
    sys.path.insert(0, os.path.dirname(_ENGINE))
    try:
        import spm_engine  # noqa: PLC0415
        engine_note = spm_engine._PARAMS_JSON["note"]
    finally:
        sys.path.pop(0)
    with open(_PARAMS_FILE, "r", encoding="utf-8") as fh:
        file_note = json.load(fh)["note"]
    assert engine_note == file_note, (
        "embedded note differs from the release file note:\n"
        f"file:   {file_note!r}\nengine: {engine_note!r}"
    )


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(
        (n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)
    ):
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
    print("=" * 60)
    if failures:
        print(f"FAILED ({failures})")
        sys.exit(1)
    print("ALL TESTS PASSED")
    sys.exit(0)
