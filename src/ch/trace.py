"""Trace：每步的 token / 延迟 / 成本 / 工具调用记录。

这是"可观测性"的落地点，也是对外披露数字的来源：
没有 trace，就只能说"我做了个 agent"；有 trace，才能说"token/任务降了 34%"。
"""
from __future__ import annotations

import json
from pathlib import Path

from .types import StepRecord


class Tracer:
    def __init__(self) -> None:
        self.records: list[StepRecord] = []
        self.notes: list[str] = []

    def add(self, rec: StepRecord) -> None:
        self.records.append(rec)

    def note(self, text: str) -> None:
        self.notes.append(text)

    # ------------------------------------------------------------- aggregates
    @property
    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.records)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self.records)

    @property
    def total_completion_tokens(self) -> int:
        return sum(r.completion_tokens for r in self.records)

    @property
    def total_latency_ms(self) -> float:
        return sum(r.latency_ms for r in self.records)

    def tool_histogram(self) -> dict[str, int]:
        h: dict[str, int] = {}
        for r in self.records:
            for t in r.tools:
                h[t] = h.get(t, 0) + 1
        return dict(sorted(h.items(), key=lambda kv: -kv[1]))

    def summary(self) -> dict:
        n = max(1, len(self.records))
        return {
            "steps": len(self.records),
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "tokens_per_step": round((self.total_prompt_tokens + self.total_completion_tokens) / n, 1),
            "cost_usd": round(self.total_cost, 6),
            "latency_ms": round(self.total_latency_ms, 1),
            "tools": self.tool_histogram(),
        }

    # ------------------------------------------------------------- rendering
    def dumps(self) -> str:
        return json.dumps(
            {"summary": self.summary(), "notes": self.notes, "steps": [r.to_dict() for r in self.records]},
            ensure_ascii=False,
            indent=2,
        )

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.dumps(), encoding="utf-8")
        return p

    def table(self) -> str:
        """终端可读的每步表格，演示时直接贴。"""
        lines = [f"{'step':>4} {'model':<16} {'in':>6} {'out':>5} {'usd':>9} {'ms':>7}  tools"]
        for r in self.records:
            lines.append(
                f"{r.step:>4} {r.model[:16]:<16} {r.prompt_tokens:>6} {r.completion_tokens:>5} "
                f"{r.cost_usd:>9.6f} {r.latency_ms:>7.1f}  {','.join(r.tools) or '-'}"
            )
        s = self.summary()
        lines.append(
            f"{'':>4} {'TOTAL':<16} {s['prompt_tokens']:>6} {s['completion_tokens']:>5} "
            f"{s['cost_usd']:>9.6f} {s['latency_ms']:>7.1f}"
        )
        return "\n".join(lines)
