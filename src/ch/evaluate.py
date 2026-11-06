"""评估 harness：任务集执行 + 指标 + 消融对照。

要回答的问题："你怎么证明你的优化真的有用？" —— 靠这个文件。

指标定义（README 里也写了，用于对外披露的指标）：
    pass_rate        任务通过率（checker 判定）
    tokens_per_task  单任务平均 token（prompt + completion）
    cost_per_task    单任务平均成本（USD，按 config.price_per_mtok 折算）
    avg_steps        平均步数
    stop_reasons     停机原因分布（max_steps 高说明预算/策略有问题）

消融（ablation）是本项目最值得展示的部分：
    baseline  大预算、无记忆、无压缩 —— 朴素实现
    budget    预算裁剪开，无记忆
    full      budget + 三层记忆 + 工具输出压缩
同一任务集跑三档，才谈得上"降了 34%"这种数字。

离线模式：tasks.json 里每个任务带 script，用 ScriptedLLM 回放。
它的作用是**验证 harness 管线**（预算/压缩/记忆/指标链路），不是模型能力评测；
真实数字必须用 --online 接真实模型跑（README 说明）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .agent import Agent
from .config import Config
from .embedding import make_embedder
from .llm import Router, ScriptedLLM
from .memory import MemoryManager
from .tools import default_registry
from .trace import Tracer


# ------------------------------------------------------------------ tasks
@dataclass
class TaskItem:
    id: str
    question: str
    checker: dict[str, Any]
    script: list[dict[str, Any]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


def load_tasks(path: str | Path) -> list[TaskItem]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        TaskItem(
            id=t["id"],
            question=t["question"],
            checker=t["checker"],
            script=t.get("script") or [],
            tags=t.get("tags") or [],
        )
        for t in data["tasks"]
    ]


def check(answer: str, spec: dict[str, Any]) -> bool:
    kind = spec.get("type", "contains")
    a = answer or ""
    if kind == "contains":
        return str(spec["value"]).lower() in a.lower()
    if kind == "exact":
        return a.strip() == str(spec["value"]).strip()
    if kind == "regex":
        return re.search(spec["value"], a) is not None
    if kind == "numeric":
        nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", a.replace(",", ""))]
        targets = [float(x) for x in spec["value"]]
        tol = float(spec.get("tol", 1e-6))
        return all(any(abs(n - t) <= tol for n in nums) for t in targets)
    if kind == "all":
        return all(check(a, s) for s in spec["value"])
    if kind == "min_len":
        # 真实模型给自由文本，没法用关键词判对错；在线跑时退化成"是否给出了实质回答"。
        # 这正是离线 checker 的局限，别把这种通过率当成能力指标。
        return len(a.strip()) >= int(spec["value"])
    raise ValueError(f"未知 checker 类型：{kind}")


# ------------------------------------------------------------------ metrics
@dataclass
class TaskOutcome:
    task_id: str
    passed: bool
    steps: int
    tokens: int
    cost_usd: float
    stop_reason: str
    answer: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "passed": self.passed,
            "steps": self.steps,
            "tokens": self.tokens,
            "cost_usd": round(self.cost_usd, 6),
            "stop_reason": self.stop_reason,
            "answer": self.answer[:300],
        }


@dataclass
class Metrics:
    variant: str
    outcomes: list[TaskOutcome] = field(default_factory=list)
    wall_ms: float = 0.0

    @property
    def n(self) -> int:
        return len(self.outcomes)

    @property
    def pass_rate(self) -> float:
        return sum(o.passed for o in self.outcomes) / self.n if self.n else 0.0

    @property
    def tokens_per_task(self) -> float:
        return sum(o.tokens for o in self.outcomes) / self.n if self.n else 0.0

    @property
    def cost_per_task(self) -> float:
        return sum(o.cost_usd for o in self.outcomes) / self.n if self.n else 0.0

    @property
    def avg_steps(self) -> float:
        return sum(o.steps for o in self.outcomes) / self.n if self.n else 0.0

    def stop_reasons(self) -> dict[str, int]:
        h: dict[str, int] = {}
        for o in self.outcomes:
            h[o.stop_reason] = h.get(o.stop_reason, 0) + 1
        return h

    def summary(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "tasks": self.n,
            "pass_rate": round(self.pass_rate, 4),
            "tokens_per_task": round(self.tokens_per_task, 1),
            "cost_per_task": round(self.cost_per_task, 6),
            "avg_steps": round(self.avg_steps, 2),
            "stop_reasons": self.stop_reasons(),
            "wall_ms": round(self.wall_ms, 1),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"summary": self.summary(), "outcomes": [o.to_dict() for o in self.outcomes]}


# ------------------------------------------------------------------ variants
def make_config(variant: str) -> Config:
    cfg = Config.from_env()
    if variant == "baseline":
        # 朴素实现：不裁剪、不记忆、不压缩
        cfg.context_budget = 200_000
        cfg.db_path = ".artifacts/eval-baseline.sqlite3"
    elif variant == "budget":
        cfg.context_budget = 2500
        cfg.db_path = ".artifacts/eval-budget.sqlite3"
    elif variant == "full":
        cfg.context_budget = 2500
        cfg.db_path = ".artifacts/eval-full.sqlite3"
    else:
        raise ValueError(f"未知 variant：{variant}")
    cfg.artifact_dir = f".artifacts/{variant}"
    cfg.ensure_dirs()
    return cfg


def build_agent(variant: str, cfg: Config, llm: Any) -> Agent:
    router = Router(cfg, main=llm, utility=llm)
    memory = None
    if variant == "full":
        # 配了 CH_EMBEDDING_MODEL 就用真实向量（带缓存与熔断），否则本地哈希
        memory = MemoryManager(cfg.db_path, context_budget=cfg.context_budget,
                               embedder=make_embedder(cfg))
    tools = default_registry(root=".", approval=lambda n, a: False, whitelist={"calculator", "list_dir", "read_file"})
    return Agent(router=router, tools=tools, cfg=cfg, memory=memory, tracer=Tracer())


# ------------------------------------------------------------------ runner
def run_variant(
    variant: str,
    tasks: list[TaskItem],
    llm_factory: Callable[[TaskItem], Any] | None = None,
) -> Metrics:
    cfg = make_config(variant)
    # 每次评测从干净的库开始，否则上一轮的记忆会污染本轮的召回指标
    db = Path(cfg.db_path)
    if db.exists():
        db.unlink()
    m = Metrics(variant=variant)
    t0 = time.perf_counter()
    for item in tasks:
        llm = llm_factory(item) if llm_factory else _online_llm(cfg)
        agent = build_agent(variant, cfg, llm)
        try:
            res = agent.run(item.question)
        finally:
            if agent.memory is not None:
                agent.memory.close()
        m.outcomes.append(
            TaskOutcome(
                task_id=item.id,
                passed=check(res.answer, item.checker),
                steps=res.steps,
                tokens=res.usage.total,
                cost_usd=res.cost_usd,
                stop_reason=res.stop_reason,
                answer=res.answer,
            )
        )
    m.wall_ms = (time.perf_counter() - t0) * 1000
    return m


def _online_llm(cfg: Config) -> Any:
    from .llm import OpenAICompatLLM

    return OpenAICompatLLM(cfg, role="main")


def compare(results: list[Metrics]) -> str:
    """把多档消融结果排成对照表。"""
    head = f"{'variant':<10} {'pass%':>7} {'tok/task':>9} {'usd/task':>10} {'steps':>6}  stop_reasons"
    lines = [head, "-" * len(head)]
    for m in results:
        s = m.summary()
        lines.append(
            f"{s['variant']:<10} {s['pass_rate']*100:>6.1f}% {s['tokens_per_task']:>9.1f} "
            f"{s['cost_per_task']:>10.6f} {s['avg_steps']:>6.2f}  {s['stop_reasons']}"
        )
    if len(results) >= 2:
        b, f = results[0], results[-1]
        if b.tokens_per_task > 0:
            delta = (f.tokens_per_task - b.tokens_per_task) / b.tokens_per_task * 100
            lines.append("")
            lines.append(f"token/任务 变化：{b.tokens_per_task:.1f} -> {f.tokens_per_task:.1f} ({delta:+.1f}%)")
        if b.cost_per_task > 0:
            dcost = (f.cost_per_task - b.cost_per_task) / b.cost_per_task * 100
            lines.append(f"成本/任务 变化：{b.cost_per_task:.6f} -> {f.cost_per_task:.6f} ({dcost:+.1f}%)")
        lines.append(f"通过率 变化：{b.pass_rate*100:.1f}% -> {f.pass_rate*100:.1f}%")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ContextHarness 评测")
    ap.add_argument("--tasks", default="tasks/tasks.json")
    ap.add_argument("--variants", nargs="+", default=["baseline", "budget", "full"])
    ap.add_argument("--online", action="store_true", help="接真实模型（需要 .env），否则用 script 回放")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    tasks = load_tasks(args.tasks)
    factory = None
    if not args.online:
        factory = lambda item: ScriptedLLM(item.script)  # noqa: E731

    results = [run_variant(v, tasks, factory) for v in args.variants]
    print(compare(results))

    out = args.out or f".artifacts/eval-{int(time.time())}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(
        json.dumps({"results": [r.to_dict() for r in results]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n明细已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
