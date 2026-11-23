"""Verify the 4K calibration profile through the published API and CLI."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from spm_rating import __version__, load_model, rate_content, rate_file, rate_notes
from spm_rating import engine
from spm_rating._core.parser import parse_content
from spm_rating.api import MODEL_NAMES, align


def chart(keys, fractional=False):
    header = f"osu file format v14\n[General]\nMode:3\n[Difficulty]\nCircleSize:{keys}\nOverallDifficulty:7\n[HitObjects]\n"
    rows = []
    for n in range(48):
        col = n % keys
        x = int((col + 0.5) * 512 / keys)
        t = 1000 + 125 * n
        if fractional and n % 2:
            t += 0.25
        rows.append(f"{x},192,{t},1,0,0:0:0:0:")
    return header + "\n".join(rows)


class CalibrationProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = load_model("detail")
        cls.patch = load_model("detail-4k")

    def test_manifest_and_opt_in_profile(self):
        m = json.loads((ROOT / "spm_version.json").read_text(encoding="utf-8"))
        self.assertEqual(m["version"], __version__)
        self.assertEqual(m["model_variants"], list(MODEL_NAMES))
        self.assertEqual(m["default_model"], "detail")
        self.assertEqual(dict(self.base.parameters), dict(self.patch.parameters))
        self.assertEqual(self.base.role_price, self.patch.role_price)
        self.assertNotEqual(self.base.identity, self.patch.identity)

    def test_4k_reference_is_the_only_change(self):
        for keys in (4, 6, 7):
            with self.subTest(keys=keys):
                raw = rate_content(chart(keys), self.patch, include_curve=True)
                old = rate_content(chart(keys), self.base, scale="reference", include_curve=True)
                self.assertEqual(raw["score"], old["raw_score"])
                self.assertEqual(raw["raw_score"], old["raw_score"])
                np.testing.assert_array_equal(raw["query"], old["query"])
                np.testing.assert_array_equal(raw["field"], old["field"])
                if keys == 4:
                    want = align(self.patch.alignment["4"], raw["raw_score"])
                    self.assertEqual(raw["reference_score"], want)
                    self.assertEqual(raw["reference_score"], old["reference_score"])
                else:
                    self.assertEqual(raw["reference_score"], old["reference_score"])

    def test_fractional_rate_and_segments_apply_one_reference(self):
        notes = parse_content(chart(4, True), rate=1.25)
        total = rate_notes(notes, self.patch, scale="reference")
        context = rate_notes(
            notes, self.patch, start_ms=notes.time[0], end_ms=notes.time[-1] + 1,
            scope="context", scale="reference",
        )
        self.assertEqual(total["score"], context["score"])
        self.assertEqual(total["score"], rate_notes(engine.mirror(notes), self.patch, scale="reference")["score"])
        for scope in ("context", "isolated"):
            opts = dict(start_ms=1649.2, end_ms=2949.6, scope=scope, scale="reference")
            result = rate_notes(notes, self.patch, **opts)
            old = rate_notes(notes, self.base, **opts)
            self.assertEqual(result["raw_score"], old["raw_score"])
            self.assertEqual(result["heads"], old["heads"])
            self.assertEqual(result["score"], align(self.patch.alignment["4"], result["raw_score"]))

    def test_cli_profile_csv_and_failure_are_independent(self):
        with tempfile.TemporaryDirectory(prefix="spm-4k-cli-") as d:
            root = Path(d)
            source = root / "calibration.osu"
            source.write_text(chart(4), encoding="utf-8")
            target = root / "scores.csv"
            default = subprocess.run([sys.executable, "-m", "spm_rating", str(source)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(default.returncode, 0, default.stderr)
            old = json.loads(default.stdout)
            self.assertEqual(old["model"], "detail")
            result = subprocess.run(
                [sys.executable, "-m", "spm_rating", str(source), str(root / "missing.osu"), "--model", "detail-4k", "--csv", str(target)],
                cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            new, failed = map(json.loads, result.stdout.splitlines())
            expected = rate_file(source, self.patch, scale="reference")
            self.assertEqual(new, expected)
            self.assertIn("error", failed)
            self.assertIn("detail-4k", target.read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    unittest.main()
