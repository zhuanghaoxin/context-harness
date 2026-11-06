"""Token 估算。

刻意不用 tiktoken：估算误差对"预算裁剪"这个用途是可接受的，
而零依赖让项目在任何机器上 `python examples/run_mock.py` 就能跑。

规则：CJK 字符 ≈ 1 token/字，其余 ≈ 1 token/4 字符。
对中英混排的实际偏差约 ±15%，README 里有说明。
"""
from __future__ import annotations

import math
import re
from typing import Iterable

_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")

# 每条消息的固定开销：role/分隔符等
_MESSAGE_OVERHEAD = 4
# 每次 tool call 的固定开销（id + name + json 括号）
_TOOL_CALL_OVERHEAD = 8


def estimate(text: str) -> int:
    """估算一段文本的 token 数。"""
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    other = len(text) - cjk
    return max(1, cjk + math.ceil(other / 4))


def estimate_messages(messages: Iterable[object]) -> int:
    """估算消息列表的 token 数（含工具调用结构开销）。"""
    total = 0
    for m in messages:
        total += estimate(getattr(m, "content", "") or "")
        total += _MESSAGE_OVERHEAD
        for _ in getattr(m, "tool_calls", None) or []:
            total += _TOOL_CALL_OVERHEAD
    return total
