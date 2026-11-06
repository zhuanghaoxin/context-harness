"""Router–Worker 双层：任务路由 + 独立上下文的 Worker 池。

要解决的问题（先讲清楚这个，再讲实现）：

    单个 Agent 循环处理多轮消息时，所有任务的上下文挤在一条历史里。
    任务 A 的工具输出会污染任务 B 的推理，且 A 没跑完时 B 只能排队。
    真实工作流里这两个假设都不成立——任务之间大多无关，应该并行。

三层职责划分（这是 Super-agent 的核心结构）：

    Filter  : 这条消息值不值得处理      -> filter.py
    Router  : 该交给哪个 Worker，还是新开一个
    Worker  : 各自独立上下文，跑自己的任务

"消息直达"是 Router 最容易被忽略的价值：
    Worker 接手之后，同一任务后续的消息直接投递给它，不再经过 Router 重建上下文。
    否则每条消息都要重新拼一次历史，token 成本随会话轮数线性上升。

与 DAG 的分工：DAG 是"我已经知道任务怎么拆"，Router 是"我还不知道该怎么拆"。
    前者确定性可复现（评测友好），后者靠过滤+相似度+模型判断，需要成本约束。
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .embedding import make_embedder, similarity
from .filter import ACT_NOW, FilterDecision, MessageFilter
from .types import RunResult, new_id
from .worker_store import WorkerRecord, WorkerStore

NEW_WORKER = "new_worker"
REUSE_WORKER = "reuse_worker"
REJECT = "reject"

# 用户显式要求拆分时，主动委派不算"主动性"
_EXPLICIT_SPLIT = ("拆", "并行", "分成", "分别", "同时", "各自", "分开做")


@dataclass
class WorkItem:
    """一条待处理的输入。enqueued_ts 是"用户发出它的时刻"，用来算端到端延迟。"""

    id: str = field(default_factory=lambda: new_id("item"))
    text: str = ""
    enqueued_ts: float = field(default_factory=time.time)
    status: str = "pending"  # pending | routed | running | done | failed | rejected
    worker_id: str | None = None
    answer: str = ""
    stop_reason: str = ""
    steps: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    started_ts: float = 0.0
    finished_ts: float = 0.0
    note: str = ""
    passed: bool | None = None

    @property
    def latency_ms(self) -> float:
        """端到端延迟（含排队）。并行是否真的省时间，只有这个指标能看出来。"""
        if not self.finished_ts:
            return 0.0
        return (self.finished_ts - self.enqueued_ts) * 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text[:120],
            "status": self.status,
            "worker_id": self.worker_id,
            "stop_reason": self.stop_reason,
            "steps": self.steps,
            "tokens": self.tokens,
            "cost_usd": round(self.cost_usd, 6),
            "latency_ms": round(self.latency_ms, 1),
            "answer": self.answer[:200],
            "note": self.note,
        }


@dataclass
class RouteDecision:
    action: str
    worker_id: str | None = None
    reason: str = ""
    proactive: bool = False
    filter_decision: FilterDecision | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "worker_id": self.worker_id,
            "reason": self.reason,
            "proactive": self.proactive,
            "filter": self.filter_decision.to_dict() if self.filter_decision else None,
        }


class Worker:
    """一个独立上下文的执行单元。上下文就是它自己那份 Agent 的 history。

    Worker 现在只是一个"把手"：真正的进度在 checkpoint（run_id = worker.id），
    状态在 worker_store。所以它可以被别的线程、别的进程接手。
    """

    def __init__(self, task: str, agent: Any, item_id: str = "", vec: list[float] | None = None):
        self.id = new_id("w")
        self.task = task
        self.agent = agent
        self.item_id = item_id
        self.status = "idle"  # idle | running | done | failed
        self.result: RunResult | None = None
        self.error = ""
        self.created_ts = time.time()
        self.messages: list[str] = []  # 后续直达消息，用于衡量"消息直达"收益
        self.pending: list[str] = []  # 已直达、还没被消费的补充消息（按序消费，不丢不重）
        self.reserved = False  # 已被 Router 认领，避免两条消息抢同一个 Worker
        self.vec = vec  # 任务向量，由 WorkerPool 用同一个 embedder 算，保证可比
        self._run_lock = threading.Lock()  # 同一个 Worker 上的多次运行必须串行

    def can_take(self, score: float, threshold: float) -> bool:
        """是否还是同一个话题。已完成的 Worker 不再接手。"""
        if self.status in ("done", "failed") or self.reserved:
            return False
        return score >= threshold

    def deliver(self, text: str) -> None:
        """消息直达：排进该 Worker 的待消费队列，下一次 run 接在已有上下文后面。

        旧实现只把文本拼进 task，而 find_reusable 又跳过 running —— 这条分支永远
        走不到；即便走到，消息也只会改任务文本、没有任何执行者。
        """
        self.messages.append(text)
        self.pending.append(text)

    def run(self, resume: bool = False) -> RunResult:
        with self._run_lock:
            self.reserved = False
            self.status = "running"
            extra = None
            if resume and self.pending:
                extra = self.pending.pop(0)
            try:
                # run_id=self.id：Agent 的每步 checkpoint 都记在这个 id 下，
                # 于是"续跑"只需要一个 worker_id，不需要别的状态传递。
                self.result = self.agent.run(self.task, run_id=self.id, resume=resume, extra=extra)
                self.status = "done"
            except Exception as e:  # noqa: BLE001 - 一个 Worker 挂掉不能拖垮整批
                self.error = f"{type(e).__name__}: {e}"
                self.status = "failed"
            return self.result  # type: ignore[return-value]

    def to_state(self) -> dict[str, Any]:
        """可序列化的最小状态，用于存算分离（主进程只持 worker_id）。"""
        return {
            "worker_id": self.id,
            "task": self.task,
            "status": self.status,
            "messages": list(self.messages),
            "created_ts": self.created_ts,
        }


class WorkerPool:
    """Worker 的容器 + 异步执行器 + 持久化入口。

    三个能力对应三件事：
        spawn   -> 上下文隔离（每个 Worker 一个全新 Agent）
        submit  -> 异步执行（提交后立刻返回 worker_id，不阻塞调用方）
        resume  -> 存算分离（进程挂了，凭 id 从 checkpoint 续跑）
    """

    def __init__(
        self,
        cfg: Any,
        agent_factory: Callable[[], Any],
        max_workers: int | None = None,
        store: Any = None,
        embedder: Any = None,
    ):
        self.cfg = cfg
        self.agent_factory = agent_factory
        self.max_workers = max_workers or getattr(cfg, "max_workers", 4)
        self.workers: dict[str, Worker] = {}
        self._lock = threading.Lock()
        self.peak_concurrency = 0
        self._inflight = 0
        self._store = store
        self._embedder = embedder
        self._executor: Any = None
        self._futures: dict[str, Any] = {}

    # ------------------------------------------------------------ 依赖（懒加载）
    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            self._embedder = make_embedder(self.cfg)
        return self._embedder

    @property
    def store(self) -> Any:
        if self._store is None:
            art = getattr(self.cfg, "artifact_dir", ".artifacts")
            self._store = WorkerStore(Path(str(art)) / "workers.sqlite3")
        return self._store

    @property
    def executor(self) -> Any:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="ch-worker")
        return self._executor

    # ---------------------------------------------------------------- pool
    def spawn(self, task: str, item_id: str = "") -> Worker:
        # agent_factory 每次返回全新 Agent，保证 Worker 之间上下文不共享
        vec = self.embedder.embed(task)
        w = Worker(task=task, agent=self.agent_factory(), item_id=item_id, vec=vec)
        with self._lock:
            self.workers[w.id] = w
        return w

    @property
    def active(self) -> list[Worker]:
        return [w for w in self.workers.values() if w.status in ("idle", "running")]

    def score(self, text: str, worker: Worker) -> float:
        """话题相似度。用向量余弦而不是字面 n-gram 重合，才能认出同义改写。"""
        if worker.vec is None:
            worker.vec = self.embedder.embed(worker.task)
        return similarity(self.embedder.embed(text), worker.vec)

    def find_reusable(self, text: str, threshold: float) -> Worker | None:
        best, best_score = None, 0.0
        query = self.embedder.embed(text)  # 只算一次，别在循环里重复调远端
        for w in self.active:
            if w.vec is None:
                w.vec = self.embedder.embed(w.task)
            if not w.can_take(similarity(query, w.vec), threshold):
                continue
            s = similarity(query, w.vec)
            if s > best_score:
                best, best_score = w, s
        return best

    # -------------------------------------------------------- 异步执行 / 存算分离
    def submit(self, task: str, item_id: str = "", batch_id: str = "") -> str:
        """提交后立刻返回 worker_id，不阻塞。调用方之后用 poll / wait。"""
        w = self.spawn(task, item_id)
        w.status = "running"
        self._persist(w, batch_id=batch_id)
        fut = self.executor.submit(self._run_worker, w, batch_id)
        with self._lock:
            self._futures[w.id] = fut
        return w.id

    def _run_worker(self, w: Worker, batch_id: str = "") -> None:
        with self._lock:
            self._inflight += 1
            self.peak_concurrency = max(self.peak_concurrency, self._inflight)
        try:
            w.run()
        finally:
            with self._lock:
                self._inflight -= 1
            self._persist(w, batch_id=batch_id)

    def poll(self, worker_id: str) -> dict[str, Any] | None:
        """从（可能的）另一个进程写过的库里读状态——主进程不需要持有 Worker 对象。"""
        rec = self.store.load(worker_id)
        return rec.to_dict() if rec else None

    def wait(self, worker_id: str, timeout: float | None = None) -> Worker | None:
        fut = self._futures.get(worker_id)
        if fut is not None:
            try:
                fut.result(timeout=timeout)
            except Exception:  # noqa: BLE001 - 失败信息已记在 Worker.error 里
                pass
        return self.workers.get(worker_id)

    def resume(self, worker_id: str) -> str:
        """凭 id 续跑：进度来自 checkpoint，任务文本来自 worker_store。

        注意"done"不等于"跑完了"：被 max_steps / watchdog 截断的 run 状态也是 done，
        但它们没给出最终答案，续跑才有意义。只有 stop_reason == final 才算真结束。
        """
        rec = self.store.load(worker_id)
        if rec is None:
            raise KeyError(f"unknown worker_id: {worker_id}")
        if rec.status == "done" and rec.stop_reason == "final":
            return worker_id
        w = Worker(task=rec.text, agent=self.agent_factory(), vec=self.embedder.embed(rec.text))
        w.id = worker_id
        w.status = "running"
        with self._lock:
            self.workers[worker_id] = w
        self._persist(w, batch_id=rec.batch_id)
        fut = self.executor.submit(self._run_worker_resume, w, rec.batch_id)
        with self._lock:
            self._futures[worker_id] = fut
        return worker_id

    def _run_worker_resume(self, w: Worker, batch_id: str = "") -> None:
        with self._lock:
            self._inflight += 1
            self.peak_concurrency = max(self.peak_concurrency, self._inflight)
        try:
            w.run(resume=True)
        finally:
            with self._lock:
                self._inflight -= 1
            self._persist(w, batch_id=batch_id)

    def _persist(self, w: Worker, batch_id: str = "") -> None:
        res = w.result
        rec = WorkerRecord(
            worker_id=w.id,
            text=w.task,
            status=w.status,
            run_id=w.id,
            answer=(res.answer if res else ""),
            stop_reason=(res.stop_reason if res else ""),
            steps=(res.steps if res else 0),
            tokens=((res.usage.prompt_tokens + res.usage.completion_tokens) if res else 0),
            cost_usd=(res.cost_usd if res else 0.0),
            error=w.error,
            batch_id=batch_id,
            created_ts=w.created_ts,
        )
        self.store.save(rec)

    def snapshot(self) -> list[dict[str, Any]]:
        return [w.to_state() for w in self.workers.values()]

    def close(self) -> None:
        """关闭执行器、Worker 背后的 sqlite、以及状态库。"""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        for w in self.workers.values():
            mem = getattr(w.agent, "memory", None)
            if mem is not None:
                try:
                    mem.close()
                except Exception:
                    pass
        if self._store is not None:
            self._store.close()
            self._store = None


class TaskRouter:
    """决定一条消息的去向。规则优先，模型只在灰区被问一次。"""

    def __init__(self, cfg: Any, pool: WorkerPool, msg_filter: MessageFilter | None = None):
        self.cfg = cfg
        self.pool = pool
        self.filter = msg_filter or MessageFilter(cfg)
        self.reuse_threshold = getattr(cfg, "worker_reuse_threshold", 0.35)
        self.queue_factor = getattr(cfg, "queue_factor", 3)
        self.decisions: list[RouteDecision] = []

    def route(self, text: str, item_id: str = "") -> RouteDecision:
        fd = self.filter.decide(text)

        if fd.action != ACT_NOW:
            d = RouteDecision(REJECT, reason=f"filter:{fd.action}", filter_decision=fd)
            self.decisions.append(d)
            return d

        # 1) 同一话题仍在进行 -> 消息直达，不重建上下文
        reusable = self.pool.find_reusable(text, self.reuse_threshold)
        if reusable is not None:
            reusable.deliver(text)
            reusable.reserved = True
            d = RouteDecision(
                REUSE_WORKER,
                worker_id=reusable.id,
                reason=f"话题相似度 >= {self.reuse_threshold}",
                filter_decision=fd,
            )
            self.decisions.append(d)
            return d

        # 2) 硬背压：不在 max_workers 处拒收（排队是正常的），只有待处理消息
        #    堆到 queue_factor 倍槽位才拒收，避免无界内存增长。
        #    max_workers 是"同时跑几个"，队列是"还能接多少"，两者不能混成一个数。
        queue_limit = max(1, int(self.pool.max_workers * self.queue_factor))
        if len(self.pool.active) >= queue_limit:
            d = RouteDecision(REJECT, reason=f"队列已满({queue_limit})", filter_decision=fd)
            self.decisions.append(d)
            return d

        # 3) 新开 Worker。"主动性"= 用户没要求拆分也主动分了工
        proactive = not any(k in text for k in _EXPLICIT_SPLIT)
        w = self.pool.spawn(text, item_id)
        d = RouteDecision(
            NEW_WORKER,
            worker_id=w.id,
            reason="新任务，独立上下文",
            proactive=proactive,
            filter_decision=fd,
        )
        self.decisions.append(d)
        return d

    @property
    def proactive_count(self) -> int:
        return sum(1 for d in self.decisions if d.action == NEW_WORKER and d.proactive)

    def stats(self) -> dict[str, Any]:
        by: dict[str, int] = {}
        for d in self.decisions:
            by[d.action] = by.get(d.action, 0) + 1
        return {
            "decisions": len(self.decisions),
            "by_action": by,
            "proactive": self.proactive_count,
            "filter": self.filter.stats.to_dict(),
        }
