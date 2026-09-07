"""按 SPM 标准化方案执行接口合规检查，输出 compliance_report.json。"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTERFACE_DIR = os.path.join(REPO_ROOT, "interface", "python")
MANIFEST_PATH = os.path.join(REPO_ROOT, "interface", "manifest.json")
EXPECTED_PATH = os.path.join(REPO_ROOT, "tests", "expected.json")
CASES_DIR = os.path.join(REPO_ROOT, "tests", "cases")
OUT_PATH = os.path.join(REPO_ROOT, "tests", "compliance_report.json")

PY_EXE = sys.executable

report = {"checks": [], "summary": {}}


def record(name, passed, detail=""):
    report["checks"].append({"name": name, "passed": bool(passed),
                             "detail": detail})
    print("[%s] %s: %s" % ("PASS" if passed else "FAIL", name, detail))


# (a) interface/python 复制到空目录后可独立运行：path 模式 + content 模式
with tempfile.TemporaryDirectory(prefix="spm_iso_") as tmpdir:
    for fname in os.listdir(INTERFACE_DIR):
        src = os.path.join(INTERFACE_DIR, fname)
        if os.path.isfile(src):
            shutil.copy(src, tmpdir)

    sample_osu = os.path.join(CASES_DIR, sorted(os.listdir(CASES_DIR))[0])

    code = (
        "import sys, json; sys.path.insert(0, r'{tmp}');"
        "import spm_interface as spm;"
        "r1 = spm.analyze(r'{p}', input_type='path');"
        "c = open(r'{p}', encoding='utf-8').read();"
        "r2 = spm.analyze(c, input_type='content');"
        "print(json.dumps({{'path': r1, 'content': r2}}))"
    ).format(tmp=tmpdir, p=sample_osu)

    proc = subprocess.run([PY_EXE, "-c", code], capture_output=True,
                          text=True, timeout=300)
    ok = proc.returncode == 0
    detail = ""
    if ok:
        try:
            out = json.loads(proc.stdout.strip())
            sp = out["path"].get("star")
            sc = out["content"].get("star")
            ok = (isinstance(sp, (int, float)) and sp > 0
                  and isinstance(sc, (int, float)) and sc > 0)
            detail = "path star=%s, content star=%s" % (sp, sc)
        except Exception as e:
            ok = False
            detail = "parse output failed: %s; stdout=%r; stderr=%r" % (
                e, proc.stdout, proc.stderr)
    else:
        detail = "subprocess failed: stderr=%r" % (proc.stderr,)
    record("(a) interface/python 复制到空目录后独立运行 (path + content)",
           ok, detail)


# 后续检查在仓库内做（直接 import）
sys.path.insert(0, INTERFACE_DIR)
import spm_interface as spm  # noqa: E402

sample_osu = os.path.join(CASES_DIR, sorted(os.listdir(CASES_DIR))[0])

# (b) full 模式键集合完整，未实现字段为哨兵
full = spm.analyze(sample_osu, mode="full")
expected_keys = {
    "schema_version", "algorithm", "star", "star_rc", "star_ln",
    "sort", "tags", "curves", "performance", "extras",
}
actual_keys = set(full.keys())
keys_match = actual_keys == expected_keys

sentinel_ok = (
    full.get("star_rc") == -255
    and full.get("star_ln") == -255
    and full.get("sort") == ""
    and full.get("tags") is None
    and full.get("curves") is None
    and full.get("extras") == {}
    and isinstance(full.get("performance"), dict)
    and "compute_time_ms" in full.get("performance", {})
)
record(
    "(b) full 模式键集合完整，未实现字段为哨兵",
    keys_match and sentinel_ok,
    "keys_match=%s actual=%s sentinel_ok=%s" % (
        keys_match, sorted(actual_keys), sentinel_ok),
)

# (c) capabilities() 与 manifest.json 一致
with open(MANIFEST_PATH, encoding="utf-8") as f:
    manifest = json.load(f)
caps_runtime = spm.capabilities()
caps_manifest = manifest.get("capabilities")
record(
    "(c) capabilities() 与 manifest.json 一致",
    caps_runtime == caps_manifest,
    "runtime=%s manifest=%s" % (caps_runtime, caps_manifest),
)

# (d) expected.json 重跑容差内一致
with open(EXPECTED_PATH, encoding="utf-8") as f:
    expected = json.load(f)
tol = expected.get("tolerance", 0.001)
all_ok = True
details = []
for case in expected["cases"]:
    fname = os.path.basename(case["file"])
    fpath = os.path.join(CASES_DIR, fname)
    res = spm.analyze(fpath, input_type="path",
                      speed_rate=case.get("speed_rate", 1.0))
    got = res.get("star", -255)
    want = case["star"]
    ok = abs(got - want) <= tol
    details.append("%s: got=%.6f want=%.6f diff=%.2e" % (
        fname, got, want, abs(got - want)))
    all_ok = all_ok and ok
record("(d) tests/expected.json 重跑容差内一致", all_ok, "; ".join(details))

# (e) 错误输入返回 error 对象而非抛异常
try:
    bad = spm.analyze("this is not an osu file content at all",
                      input_type="content")
    err_ok = isinstance(bad, dict) and bad.get("star") == -255 \
        and "error" in bad
    detail = "returned keys=%s" % (list(bad.keys()),)
except Exception as e:
    err_ok = False
    detail = "raised: %s" % (e,)
record("(e) 错误输入返回 error 对象而非抛异常", err_ok, detail)


# 汇总
passed_count = sum(1 for c in report["checks"] if c["passed"])
total = len(report["checks"])
report["summary"] = {
    "passed": passed_count,
    "total": total,
    "all_passed": passed_count == total,
}

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print("\n%d/%d checks passed. Wrote %s" % (passed_count, total, OUT_PATH))
