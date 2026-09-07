"""SPM Rating v1.0.0 —— SPM 标准化接口实现（interface_spec_version 1.0）。

自包含模块：除 numpy、scipy 外零第三方依赖。将本目录（interface/python/）
复制到任意空目录后可直接运行。

  from spm_interface import analyze, capabilities
  result = analyze(source, input_type="auto", speed_rate=1.0, mode="full")

参数说明：
- source：.osu 文件路径字符串（input_type="path"）或完整文件文本内容
  （input_type="content"）；input_type="auto" 时按规则自动判别
  （含有 "osu file format" 则视为文本内容）。
- speed_rate：速度倍率。本实现仅支持 1.0，传入其他值时仍按 1.0 计算，
  并在 full 模式的 extras 中标记 speed_rate_ignored=true。
- mode="minimal"：返回 {"schema_version", "star"}。
- mode="full"：返回完整 Schema。star_rc、star_ln 哨兵值为 -255，
  sort 哨兵值为空字符串，tags、curves 哨兵值为 null，performance
  含 compute_time_ms。
- 计算失败返回 {"schema_version": "1.0", "star": -255, "error": ...}。
"""

import os
import tempfile
import time

from rating_engine import rate_file as _rate_file

_SCHEMA_VERSION = "1.0"
_ALGORITHM = {
    "id": "spm-rating",
    "name": "SPM Rating",
    "version": "1.0.0",
    "variant": "main",
}
_CAPABILITIES = {
    "star": True,
    "star_rc": False,
    "star_ln": False,
    "sort": False,
    "tags": False,
    "curves": False,
    "performance": True,
    "extras": ["speed_rate_ignored"],
}


def capabilities():
    """返回与 manifest.json 一致的能力表。"""
    return dict(_CAPABILITIES)


def _detect_input_type(source):
    """自动判别输入类型。"""
    if not isinstance(source, str):
        return "parsed"
    if "osu file format" in source:
        return "content"
    if source.endswith(".osu") and "\n" not in source:
        return "path"
    # 字符串但既不是文本内容也不是路径——尝试按路径处理（可能报错）。
    return "path"


def analyze(source, input_type="auto", speed_rate=1.0, mode="minimal",
            options=None):
    """统一入口，语义见模块文档。"""
    start = time.perf_counter()

    if input_type == "auto":
        input_type = _detect_input_type(source)

    speed_ignored = False
    try:
        rate = float(speed_rate)
    except Exception:
        rate = 1.0
    if abs(rate - 1.0) > 1e-9:
        speed_ignored = True

    try:
        if input_type == "path":
            star = _rate_file(source)
        elif input_type == "content":
            star = _self_check_tmpfile(source)
        elif input_type == "parsed":
            raise ValueError("parsed input type is not supported by this implementation")
        else:
            raise ValueError("unknown input_type: %s" % (input_type,))
    except Exception as e:
        return {"schema_version": _SCHEMA_VERSION, "star": -255,
                "error": str(e)}

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    star = float(star)

    if mode == "full":
        extras = {}
        if speed_ignored:
            extras["speed_rate_ignored"] = True
        return {
            "schema_version": _SCHEMA_VERSION,
            "algorithm": dict(_ALGORITHM),
            "star": star,
            "star_rc": -255,
            "star_ln": -255,
            "sort": "",
            "tags": None,
            "curves": None,
            "performance": {"compute_time_ms": elapsed_ms},
            "extras": extras,
        }
    return {"schema_version": _SCHEMA_VERSION, "star": star}


def _self_check_tmpfile(content):
    """把 .osu 内容写入临时文件再计算（content 模式内部使用）。"""
    fd, tmp_path = tempfile.mkstemp(suffix=".osu", prefix="spm_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        return _rate_file(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
