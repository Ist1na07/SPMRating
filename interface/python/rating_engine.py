"""SPM Rating v1.0.0 —— 单谱面评分引擎（薄封装）。

本模块是 interface/python/spm_engine.py（自包含单文件引擎）的薄封装，
与仓库 core/ 共用一套代码。

公开函数：rate_file / rate_content / rate_notes。
"""

from spm_engine import rate_content, rate_file, rate_notes

__all__ = ["rate_file", "rate_content", "rate_notes"]
