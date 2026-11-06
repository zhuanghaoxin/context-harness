"""Supervisor：把 Input → Filter → Router → Worker 串起来，并做模式对照。

两种模式，就是对比表里的两个实验臂：

    single  一条消息一个 loop 串行跑完。没有过滤，没有路由，没有并行。
            这就是"普通 chat agent 当工作流用"的样子。
    super   每条消息先过 Filter，再由 Router 决定：直达已有 Worker / 新开 Worker / 拒收。
            Worker 之间有独立上下文，可以并行。

**这个离线对照实验能测什么、不能测什么，必须讲清楚（评审必问）：**

能测：消息过滤省掉的无效处理、主动委派次数、并行带来的 makespan 与端到端延迟、
      并发峰値、每条消息的 token/成本。
不能测：真实模型在"该不该委派"上的判断质量——离线是脚本回放，模型行为是固定的。
      所以论文里那个 -47% 不能直接搬，我们报的是"机制在路上省掉了什么"。

真要拿真实数字，用 `--online` 换成 OpenAICompatLLM 重跑，脚本回放换成真实任务。
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .evaluate import check
from .filter import MessageFilter
from .router_worker import (
    NEW_WORKER,
    REJECT,
    REUSE_WORKER,
    RouteDecision,
    TaskRouter,
    WorkItem,
    WorkerPool,
)

MODE_SINGLE = "single"
MODE_SUPER = "super"


def _pct(values: list[float], p: float) -> float:
    """简单分位数（线性插值）。样本很少时也能给个数，比只说均值诚实。"""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


@dataclass
class Report:
    mode: str
    items: list[WorkItem] = field(default_factory=list)
    makespan_ms: float = 0.0
    cost_usd: float = 0.0
    tokens: int = 0
    peak_concurrency: int = 0
    router_stats: dict[str, Any] = field(default_factory=dict)
    filter_stats: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ 派生
    @property
    def submitted(self) -> int:
        return len(self.items)

    @property
    def processed(self) -> list[WorkItem]:
        return [i for i in self.items if i.status in ("done", "failed")]

    @property
    def completed(self) -> list[WorkItem]:
        return [i for i in self.items if i.status == "done" and i.answer]

    @property
    def rejected(self) -> list[WorkItem]:
        return [i for i in self.items if i.status == "rejected"]

    @property
    def graded(self) -> list[WorkItem]:
        return [i for i in self.items if i.passed is not None]

    @property
    def pass_rate(self) -> float:
        g = self.graded
        if not g:
            return 0.0
        return sum(1 for i in g if i.passed) / len(g)

    @property
    def latencies(self) -> list[float]:
        return [i.latency_ms for i in self.items if i.finished_ts and i.status != "rejected"]

    @property
    def p50_ms(self) -> float:
        return _pct(self.latencies, 0.50)

    @property
    def p95_ms(self) -> float:
        return _pct(self.latencies, 0.95)

    @property
    def attempts(self) -> int:
        """实际跑了 loop 的消息数（被拒的不算）。分母决定 tokens_per_task 怎么算。"""
        return len(self.processed)

    @property
    def tokens_per_task(self) -> float:
        n = len(self.completed) or self.attempts
        return self.tokens / n if n else 0.0

    @property
    def cost_per_task(self) -> float:
        n = len(self.completed) or self.attempts
        return self.cost_usd / n if n else 0.0

    @property
    def delegated(self) -> int:
        return int(self.router_stats.get("by_action", {}).get(NEW_WORKER, 0))

    @property
    def reused(self) -> int:
        return int(self.router_stats.get("by_action", {}).get(REUSE_WORKER, 0))

    @property
    def proactive(self) -> int:
        return int(self.router_stats.get("proactive", 0))

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "submitted": self.submitted,
            "processed": self.attempts,
            "completed": len(self.completed),
            "rejected": len(self.rejected),
            "pass_rate": round(self.pass_rate, 4),
            "makespan_ms": round(self.makespan_ms, 1),
            "p50_ms": round(self.p50_ms, 1),
            "p95_ms": round(self.p95_ms, 1),
            "peak_concurrency": self.peak_concurrency,
            "delegated": self.delegated,
            "reused": self.reused,
            "proactive": self.proactive,
            "tokens": self.tokens,
            "tokens_per_task": round(self.tokens_per_task, 1),
            "cost_usd": round(self.cost_usd, 6),
            "cost_per_task": round(self.cost_per_task, 6),
            "filter": self.filter_stats,
        }

    def table(self) -> str:
        d = self.summary()
        return (
            f"[{d['mode']}] 提交 {d['submitted']} / 实跑 {d['processed']} / 完成 {d['completed']} "
            f"/ 拒收 {d['rejected']} | 通过率 {d['pass_rate']:.0%}\n"
            f"  makespan {d['makespan_ms']:.0f}ms  P50 {d['p50_ms']:.0f}ms  P95 {d['p95_ms']:.0f}ms  "
            f"峰值并发 {d['peak_concurrency']}\n"
            f"  委派 {d['delegated']}（主动 {d['proactive']}）复用 {d['reused']} | "
            f"token {d['tokens']} ({d['tokens_per_task']:.1f}/任务) 成本 ${d['cost_usd']:.4f}"
        )


class Supervisor:
    """两种模式的执行器。agent_factory 每次返回全新 Agent，保证 Worker 上下文隔离。"""

    def __init__(
        self,
        cfg: Any,
        agent_factory: Callable[[], Any],
        checks: dict[str, dict[str, Any]] | None = None,
    ):
        self.cfg = cfg
        self.agent_factory = agent_factory
        self.checks = checks or {}
        self._inflight = 0
        self._peak_inflight = 0
        self._lock = threading.Lock()
        self._lock_t0 = 0.0

    # ------------------------------------------------------------ 打分
    def _grade(self, item: WorkItem) -> None:
        spec = self.checks.get(item.id)
        if not spec:
            return
        item.passed = check(item.answer, spec)

    # ------------------------------------------------------------ single
    def run_single(self, items: list[WorkItem]) -> Report:
        """串行基线：不过滤、不路由、不并行，一条接一条跑完。"""
        agent = self.agent_factory()
        report = Report(mode=MODE_SINGLE, items=items)
        t0 = time.time()
        try:
            for it in items:
                it.status = "running"
                it.started_ts = time.time()
                try:
                    res = agent.run(it.text)
                    it.answer = res.answer
                    it.stop_reason = res.stop_reason
                    it.steps = res.steps
                    it.tokens = res.usage.prompt_tokens + res.usage.completion_tokens
                    it.cost_usd = res.cost_usd
                    it.status = "done"
                except Exception as e:  # noqa: BLE001
                    it.status = "failed"
                    it.note = f"{type(e).__name__}: {e}"
                it.finished_ts = time.time()
                self._grade(it)
                report.tokens += it.tokens
                report.cost_usd += it.cost_usd
                report.peak_concurrency = max(report.peak_concurrency, 1)
        finally:
            self._close_agent(agent)
        report.makespan_ms = (time.time() - t0) * 1000
        return report

    # ------------------------------------------------------------- super
    def run_super(self, items: list[WorkItem]) -> Report:
        """双层：Filter 排噪 → Router 分流 → Worker 池并行。"""
        pool = WorkerPool(self.cfg, self.agent_factory)
        router = TaskRouter(self.cfg, pool, MessageFilter(self.cfg))
        report = Report(mode=MODE_SUPER, items=items)
        self._inflight = 0
        self._peak_inflight = 0
        t0 = time.time()
        running = 0
        try:
            with ThreadPoolExecutor(max_workers=pool.max_workers) as ex:
                futures: dict[Any, WorkItem] = {}
                for it in items:
                    decision = router.route(it.text, it.id)
                    if decision.action == REJECT:
                        it.status = "rejected"
                        it.note = decision.reason
                        it.finished_ts = time.time()
                        continue
                    it.started_ts = time.time()
                    it.status = "running"
                    # Worker 由 Router 建好（NEW_WORKER 时新建，REUSE 时复用同一个），这里只负责跑
                    w = pool.workers[decision.worker_id]  # type: ignore[index]
                    it.worker_id = w.id
                    # REUSE 的语义是"接着这个 Worker 已有的上下文继续跑"，
                    # 所以必须 resume=True，否则会把基础任务整个重做一遍。
                    fut = ex.submit(self._exec, w, it, decision.action == REUSE_WORKER)
                    futures[fut] = it
                    running += 1
                    pool.peak_concurrency = max(pool.peak_concurrency, self._busy(pool))

                for fut, it in futures.items():
                    fut.result()  # _exec 内部已吞异常，这里只等它结束
                    self._grade(it)
                    report.tokens += it.tokens
                    report.cost_usd += it.cost_usd
        finally:
            pool.close()
        report.makespan_ms = (time.time() - t0) * 1000
        report.peak_concurrency = max(pool.peak_concurrency, self._peak_inflight)
        report.router_stats = router.stats()
        report.filter_stats = router.filter.stats.to_dict()
        return report

    # ------------------------------------------------------------- 内部
    @staticmethod
    def _busy(pool: WorkerPool) -> int:
        return sum(1 for w in pool.workers.values() if w.status == "running")

    def _exec(self, worker: Any, item: WorkItem, resume: bool = False) -> None:
        """在 Worker 线程里跑一个 loop，结果写回 WorkItem。异常不外抛。"""
        with self._lock:
            self._inflight += 1
            self._peak_inflight = max(self._peak_inflight, self._inflight)
        try:
            res = worker.run(resume=resume)
            if res is None:
                item.status = "failed"
                item.note = worker.error or "worker returned None"
            else:
                item.answer = res.answer
                item.stop_reason = res.stop_reason
                item.steps = res.steps
                item.tokens = res.usage.prompt_tokens + res.usage.completion_tokens
                item.cost_usd = res.cost_usd
                item.status = "done"
        except Exception as e:  # noqa: BLE001 - 单个 Worker 失败不影响整批
            item.status = "failed"
            item.note = f"{type(e).__name__}: {e}"
        finally:
            item.finished_ts = time.time()
            with self._lock:
                self._inflight -= 1

    @staticmethod
    def _close_agent(agent: Any) -> None:
        mem = getattr(agent, "memory", None)
        if mem is not None:
            try:
                mem.close()
            except Exception:
                pass


def compare(reports: list[Report]) -> list[dict[str, Any]]:
    """对照表：把 single 的变化幅度打到每一行，直接能贴进报告/文档。"""
    if not reports:
        return []
    base = reports[0].summary()
    rows: list[dict[str, Any]] = []
    for r in reports:
        s = r.summary()
        row = dict(s)
        for k in ("makespan_ms", "p95_ms", "tokens", "tokens_per_task", "cost_usd", "pass_rate"):
            b, v = base.get(k) or 0, s.get(k) or 0
            if b:
                row[f"d_{k}"] = f"{(v - b) / b * 100:+.1f}%"
            else:
                row[f"d_{k}"] = "-"
        rows.append(row)
    return rows
