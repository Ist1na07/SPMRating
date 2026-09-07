"""生成 tests/expected.json：用 interface 计算 cases 谱面的 star。"""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTERFACE_DIR = os.path.join(REPO_ROOT, "interface", "python")
sys.path.insert(0, INTERFACE_DIR)

import spm_interface as spm  # noqa: E402

CASES_DIR = os.path.join(REPO_ROOT, "tests", "cases")
OUT_PATH = os.path.join(REPO_ROOT, "tests", "expected.json")

case_files = sorted(f for f in os.listdir(CASES_DIR) if f.endswith(".osu"))

cases = []
for fname in case_files:
    fpath = os.path.join(CASES_DIR, fname)
    res = spm.analyze(fpath, input_type="path", speed_rate=1.0, mode="minimal")
    star = res.get("star", -255)
    print("%s: star=%s" % (fname, star))
    cases.append({
        "file": "cases/%s" % fname,
        "speed_rate": 1.0,
        "star": star,
        "star_rc": -255,
        "star_ln": -255,
        "sort": "",
        "tags": None,
    })

payload = {"tolerance": 0.001, "cases": cases}
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
print("\nWrote %s" % OUT_PATH)
