"""工具输出压缩器。

问题：一次 `grep` / `cat` 可能返回 200KB，直接塞回上下文会挤掉历史、拉高成本、
并让模型在噪音里失焦（tool result 爆炸 → 上下文漂移）。

策略（三级，由轻到重）：
  1. 未超预算 → 原样返回；
  2. 超预算   → head + tail 保留 + 中间省略标记，全文归档到 artifact 目录，
                 消息里带上 ref，需要时可回查（可回溯，不是"信息丢失"）；
  3. 有 LLM   → 用 utility 小模型对归档全文做要点摘要（成本走小模型，见 llm.Router）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .budget import fit_prefix_chars, fit_suffix_chars, truncate_to_tokens
from .tokens import estimate

# ref 只允许这些字符：它会直接变成文件名，绝不能带路径分隔符或 ".."
_REF_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass
class Compressed:
    text: str
    omitted_chars: int = 0
    ref: str | None = None
    original_tokens: int = 0
    final_tokens: int = 0


class ToolOutputCompressor:
    def __init__(self, artifact_dir: str | Path = ".artifacts", head_ratio: float = 0.6):
        self.dir = Path(artifact_dir) / "tool_outputs"
        self.head_ratio = head_ratio

    @staticmethod
    def _safe_ref(name: str, call_id: str) -> str:
        """把任意工具名/调用 id 收敛成安全文件名，杜绝 ../../ 逃逸。"""
        ref = f"{name}-{call_id}"
        ref = _REF_UNSAFE.sub("_", ref).strip(".")[:120]
        return ref or "ref"

    def compress(self, call_id: str, name: str, text: str, max_tokens: int) -> Compressed:
        original = estimate(text)
        if original <= max_tokens or max_tokens <= 0:
            return Compressed(text=text, original_tokens=original, final_tokens=original)

        self.dir.mkdir(parents=True, exist_ok=True)
        ref = self._safe_ref(name, call_id)
        path = self.dir / f"{ref}.txt"
        path.write_text(text, encoding="utf-8")

        # head/tail 都按 token 反推字符数：CJK 与 ASCII 的字符/token 比差 4 倍，
        # 固定 2.5 字符/token 会让中文输出超预算（实测 100 token 装出 273）。
        out = ""
        omitted = 0
        head_n, tail_n = fit_prefix_chars(text, max(1, int(max_tokens * self.head_ratio))), 0
        for _ in range(8):
            head = text[:head_n]
            remaining = max(0, max_tokens - estimate(head))
            tail_n = fit_suffix_chars(text, remaining)
            tail = text[len(text) - tail_n :] if tail_n > 0 else ""
            omitted = len(text) - len(head) - len(tail)
            marker = (
                f"\n... [省略 {omitted} 字符 / 约 {original - estimate(head + tail)} tokens，"
                f"全文已归档，ref={ref}] ...\n"
            )
            if estimate(marker) >= max_tokens:
                # marker 本身都放不下：退化为纯裁剪，优先保证不超预算
                out = truncate_to_tokens(text, max_tokens)
                omitted = len(text) - len(out)
                break
            out = head + marker + tail
            if estimate(out) <= max_tokens:
                break
            # marker 比预留的大：等比收缩 head，下一轮重算 tail
            head_n = max(0, head_n - max(1, (estimate(out) - max_tokens) * 2))
        if not out:
            out = truncate_to_tokens(text, max_tokens)
            omitted = len(text) - len(out)
        if estimate(out) > max_tokens:
            # 兜底：head+tail+marker 仍超预算时，只保留 head + 精简 marker（ref 仍可见）
            marker = f"\n... [省略更多内容，全文已归档，ref={ref}] ...\n"
            out = truncate_to_tokens(text, max_tokens, marker=marker)
            omitted = len(text) - max(0, len(out) - len(marker))

        return Compressed(
            text=out,
            omitted_chars=omitted,
            ref=ref,
            original_tokens=original,
            final_tokens=estimate(out),
        )

    def read_ref(self, ref: str) -> str | None:
        """只允许读回本目录内的归档，ref 里出现路径成分一律拒绝。"""
        base = self.dir.resolve()
        p = (base / f"{ref}.txt").resolve()
        if p.parent != base or not p.is_file():
            return None
        return p.read_text(encoding="utf-8")
