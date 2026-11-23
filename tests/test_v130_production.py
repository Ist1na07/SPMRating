"""Self-contained v1.3.0 production checks using Python stdlib unittest (zero research/ or pytest deps)."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import csv
import json
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spm_rating import load_model, predict_notes, rate_notes, rate_content
from spm_rating import clock, engine, v130_rev2 as core
from spm_rating._core.parser import parse_file, parse_content
from spm_rating.api import align

EXAMPLE = ROOT / "examples/Control_P_120.osu"
PACKAGED = ROOT / "spm_rating/models"
PROFILES = {"detail": "V13-D1", "compact": "V13-C"}


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def run_python(*args):
    return subprocess.run(
        [sys.executable, *map(str, args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        timeout=90,
    )


def synthetic_chart(keys, ln=False, chord_size=1, step_ms=125, count=48, fixed_order=None):
    header = f"osu file format v14\n[General]\nMode:3\n[Difficulty]\nCircleSize:{keys}\nOverallDifficulty:8\n[HitObjects]\n"
    rows = []
    for n in range(count):
        t = 1000 + n * step_ms
        cols = fixed_order[:chord_size] if fixed_order is not None else [(n + k) % keys for k in range(chord_size)]
        for col in sorted(set(cols)):
            x = int((col + 0.5) * 512 / keys)
            obj = f"128,0,{t + int(step_ms * 1.5)}:0:0:0:" if ln else "1,0,0:0:0:0:"
            rows.append(f"{x},192,{t},{obj}")
    return header + "\n".join(rows)


NO_RESEARCH_SCRIPT = """
import sys
class NoResearch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'research':
            raise AssertionError(fullname)
sys.meta_path.insert(0, NoResearch())
import spm_rating
import spm_rating.gui
for name in ('detail', 'compact', 'detail-4k'):
    r = spm_rating.rate_file(sys.argv[1], name, scale='reference')
    assert r['model'] == name
    assert spm_rating.load_model(name).revision == 2
assert not any(x.split('.')[0] == 'research' for x in sys.modules)
print('zero research modules: detail, compact, detail-4k; GUI import OK')
"""


class V130ProductionTests(unittest.TestCase):
    def test_overlay_signed_identity_and_monotone_alignment(self):
        path = PACKAGED / "detail.json"
        base = read(path)
        patch = read(PACKAGED / "detail-4k.json")
        self.assertEqual(patch["revision"], 2)
        self.assertIs(patch["pure"], True)
        canonical = path.read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(patch["base_model_identity"], sha256(canonical).hexdigest())
        self.assertEqual(load_model("detail").identity, patch["base_model_identity"])
        self.assertEqual(patch["frozen_sha256"], base["frozen_sha256"])
        self.assertEqual(patch["alignment"], {"4": base["alignment"]["4"]})
        self.assertTrue(all(bool(np.all(np.asarray(a["slopes"]) > 0)) for a in base["alignment"].values()))

    def test_model_identity_is_independent_of_checkout_line_endings(self):
        names = ("detail", "compact", "detail-4k")
        with tempfile.TemporaryDirectory(prefix="spm-model-eol-") as tmp:
            lf, crlf = Path(tmp) / "lf", Path(tmp) / "crlf"
            lf.mkdir()
            crlf.mkdir()
            for name in names:
                filename = f"{name}.json"
                canonical = (PACKAGED / filename).read_bytes().replace(b"\r\n", b"\n")
                (lf / filename).write_bytes(canonical)
                (crlf / filename).write_bytes(canonical.replace(b"\n", b"\r\n"))
            for name in names:
                with self.subTest(model=name):
                    expected = load_model(name, model_dir=lf)
                    actual = load_model(name, model_dir=crlf)
                    self.assertEqual(actual.identity, expected.identity)
                    self.assertEqual(actual.parameters, expected.parameters)
                    self.assertEqual(actual.alignment, expected.alignment)

    def test_reference_uses_positive_slopes_and_legacy_power(self):
        a = dict(domain="dan", knots=[2.0, 5.0, 9.0], slopes=[0.5, 2.0], value_at_first=1.0)
        self.assertEqual([align(a, x) for x in (0.0, 2.0, 4.0, 5.0, 9.0, 10.0)], [0.0, 1.0, 2.0, 2.5, 10.5, 12.5])
        for change in (dict(knots=[2.0, 2.0, 9.0]), dict(slopes=[0.5, 0.0]), dict(slopes=[0.5, np.inf])):
            with self.assertRaisesRegex(ValueError, "knots and positive"):
                align({**a, **change}, 4.0)
        self.assertEqual(align(dict(domain="dan", scale=2.0, power=0.5), 9.0), 6.0)

    def test_clean_subprocess_has_no_research_imports(self):
        self.assertIn("zero research modules", run_python("-c", NO_RESEARCH_SCRIPT, EXAMPLE).stdout)

    def test_pure_sigmoid_monotonicity_and_homogeneity(self):
        for name in PROFILES:
            with self.subTest(model=name):
                m = load_model(name)
                self.assertEqual(m.revision, 2)
                self.assertEqual(m.candidate, PROFILES[name])
                self.assertNotIn("sig_mix", m.parameters)
                v = core.variant(m.compact)
                beta, gamma, mix, _ = v.shape(m.parameters)
                self.assertEqual(mix, 1.0)
                d = np.array([0.25, 0.5, 1.0, 2.0])
                w = np.array([1.0, 2.0, 1.0, 3.0])
                a = core.relative(d, w, beta, gamma)
                for factor in (0.25, 2.0, 9.0):
                    self.assertAlmostEqual(core.relative(factor * d, w, beta, gamma), factor * a, places=12)
                self.assertGreater(core.relative(d + [0.0, 0.2, 0.0, 0.5], w, beta, gamma), a)

    def test_pure_phase_invariance_and_fractional_clock(self):
        samples = [
            parse_file(str(EXAMPLE)),
            parse_content(synthetic_chart(4, ln=True, chord_size=2, step_ms=110, count=64)),
        ]
        for name in PROFILES:
            with self.subTest(model=name):
                m = load_model(name)
                model = core.variant(m.compact)
                v7, v4, g = model.native(m.parameters)
                for n in samples:
                    s = core.state_from_notes(n, model, m.parameters)
                    vv = v4 if s.key == 4 else v7
                    d, _ = core.field(s.a, s.b, s.t, s.key, vv, m.compact, model.hill, model.xsoft, False, g if s.key == 4 else 0.0)
                    z = []
                    for i in range(16):
                        q, h, b = core.base.layout(s.interval, s.section, s.gp, s.win_ms, i / 16.0)
                        ev = core.phase_order(q, np.flatnonzero(s.gp > 0.0))
                        z.append(core.aggregate(d, s.gp, s.amount, s.held, q, h, ev, b, s.win_ms, s.env_half, len(s.head_pos), vv, *model.shape(m.parameters)))
                    self.assertEqual(float(np.ptp(z)), 0.0)

                note = samples[0]
                r = rate_notes(note, m, include_curve=True)
                translated = replace(note, time=note.time + 0.37, ln_end=note.ln_end + 0.37)
                self.assertIsNotNone(clock.plan(translated))
                b = rate_notes(translated, m, include_curve=True)
                self.assertLessEqual(abs(b["raw_score"] - r["raw_score"]), 1e-12)
                np.testing.assert_allclose(b["field"], r["field"], rtol=0, atol=1e-12)
                mirrored = predict_notes(engine.mirror(note), m, include_curve=True)
                self.assertEqual(mirrored["raw_score"], r["raw_score"])

    def test_physical_monotonicity_controls(self):
        tap_7k = synthetic_chart(7, ln=False, chord_size=2, step_ms=110, count=64)
        ln_7k = synthetic_chart(7, ln=True, chord_size=2, step_ms=130, count=64)
        for name in PROFILES:
            with self.subTest(model=name):
                m = load_model(name)
                chord_order = (3, 2, 4, 1, 5, 0, 6)
                chord_scores = [rate_content(synthetic_chart(7, ln=False, chord_size=k, step_ms=250, count=48, fixed_order=chord_order), m)["raw_score"] for k in range(1, 8)]
                self.assertTrue(bool(np.all(np.diff(chord_scores) > 0.0)), chord_scores)
                for src in (tap_7k, ln_7k):
                    rate_scores = [rate_content(src, m, rate=r)["raw_score"] for r in (0.75, 1.0, 1.25, 1.5)]
                    self.assertTrue(bool(np.all(np.diff(rate_scores) > 0.0)), rate_scores)
                    n = parse_content(src)
                    self.assertEqual(rate_notes(engine.mirror(n), m)["raw_score"], rate_notes(n, m)["raw_score"])

    def test_context_isolated_cancellation_and_conflicts(self):
        n = parse_file(str(EXAMPLE))
        start, end = n.time[20], n.time[65] + 0.5
        for name in PROFILES:
            with self.subTest(model=name):
                m = load_model(name)
                context = predict_notes(n, m, start_ms=start, end_ms=end, scope="context", include_curve=True)
                isolated = predict_notes(n, m, start_ms=start, end_ms=end, scope="isolated", include_curve=True)
                self.assertEqual(context["heads"], isolated["heads"])
                self.assertLess(context["heads"], n.n_notes)
                self.assertNotEqual(context["raw_score"], isolated["raw_score"])
                np.testing.assert_array_equal(context["query"], isolated["query"])
                calls = iter((False, False, True))
                with self.assertRaises(InterruptedError):
                    predict_notes(n, m, cancel=lambda: next(calls, True))
                obj = replace(
                    n,
                    time=np.r_[n.time, n.time[0]],
                    col=np.r_[n.col, n.col[0]],
                    ln_end=np.r_[n.ln_end, n.ln_end[0] + 42.0],
                    is_ln=np.r_[n.is_ln, True],
                    n_notes=n.n_notes + 1,
                )
                with self.assertRaisesRegex(ValueError, "Conflicting simultaneous objects"):
                    rate_notes(obj, m)
                with self.assertRaisesRegex(ValueError, "supports"):
                    rate_notes(replace(n, key_count=5), m)

    def test_cli_csv_and_gui_profile(self):
        from spm_rating import gui
        with tempfile.TemporaryDirectory(prefix="spm-v130-cli-") as d:
            tmp = Path(d)
            for name in ("detail", "compact", "detail-4k"):
                with self.subTest(model=name):
                    self.assertIn(name, gui.MODEL_CHOICES.values())
                    out = tmp / f"{name}.csv"
                    p = run_python("-m", "spm_rating", EXAMPLE, "--model", name, "--csv", out)
                    row = json.loads(p.stdout)
                    self.assertEqual(row["model"], name)
                    self.assertEqual(row["units"], "dan_sr")
                    with out.open(encoding="utf-8-sig", newline="") as f:
                        parsed = list(csv.DictReader(f))
                    self.assertEqual(len(parsed), 1)
                    self.assertEqual(parsed[0]["model_identity"], row["model_identity"])
                    self.assertEqual(float(parsed[0]["score"]), row["score"])


if __name__ == "__main__":
    unittest.main()
