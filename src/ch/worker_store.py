"""Worker 状态的持久化：存算分离的落地。

「存算分离」在这套系统里到底改了什么（按这个顺序说明）：

    之前：Worker 是一个内存对象，Agent 的历史、状态、进度都在这个对象里。
          进程一挂，或者主进程要重启，这些全丢；也只有一个进程能碰它。

    现在：Worker 只有 **id**，它的任务文本、状态、答案、用量全部落 sqlite；
          Agent 自己的每一步也已经落在 checkpoint 里（run_id = worker_id）。
          于是主进程 / 另一个进程 / 重启后的自己，拿一个 worker_id 就能：
            读状态（poll）、等结果（wait）、从断点续跑（resume）。

为什么把 run_id 直接复用成 worker_id，而不是新建一套 checkpoint：
    Agent 主循环本来就每步落盘（checkpoint.runs），再存一份就是两份状态，
    迟早不一致。这里只存"元信息"（任务文本、状态、用量），
    真正的执行进度交给 checkpoint，职责不重叠。
"""
from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
INTERRUPTED = "interrupted"  # 进程挂了但没标失败：下次启动要能认出来并续跑


@dataclass
class WorkerRecord:
    worker_id: str
    text: str
    status: str = PENDING
    run_id: str = ""
    answer: str = ""
    stop_reason: str = ""
    steps: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    error: str = ""
    batch_id: str = ""
    created_ts: float = field(default_factory=time.time)
    updated_ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "text": self.text[:80],
            "status": self.status,
            "run_id": self.run_id,
            "stop_reason": self.stop_reason,
            "steps": self.steps,
            "tokens": self.tokens,
            "cost_usd": round(self.cost_usd, 6),
            "error": self.error[:200],
            "answer": self.answer[:120],
        }


class WorkerStore:
    """一张表就是全部的 Worker 状态。主进程只持 worker_id。"""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = str(db_path) if db_path else ":memory:"
        if db_path:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS workers(
                   worker_id TEXT PRIMARY KEY,
                   text TEXT, status TEXT, run_id TEXT, answer TEXT,
                   stop_reason TEXT, steps INTEGER, tokens INTEGER,
                   cost_usd REAL, error TEXT, batch_id TEXT,
                   created_ts REAL, updated_ts REAL)"""
        )
        self.conn.commit()

    def save(self, rec: WorkerRecord) -> None:
        rec.updated_ts = time.time()
        with self._lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO workers(
                       worker_id, text, status, run_id, answer, stop_reason,
                       steps, tokens, cost_usd, error, batch_id, created_ts, updated_ts)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    rec.worker_id,
                    rec.text,
                    rec.status,
                    rec.run_id,
                    rec.answer,
                    rec.stop_reason,
                    rec.steps,
                    rec.tokens,
                    rec.cost_usd,
                    rec.error,
                    rec.batch_id,
                    rec.created_ts,
                    rec.updated_ts,
                ),
            )
            self.conn.commit()

    def load(self, worker_id: str) -> WorkerRecord | None:
        with self._lock:
            row = self.conn.execute(
                """SELECT worker_id, text, status, run_id, answer, stop_reason,
                          steps, tokens, cost_usd, error, batch_id, created_ts, updated_ts
                   FROM workers WHERE worker_id=?""",
                (worker_id,),
            ).fetchone()
        if row is None:
            return None
        keys = [
            "worker_id", "text", "status", "run_id", "answer", "stop_reason",
            "steps", "tokens", "cost_usd", "error", "batch_id", "created_ts", "updated_ts",
        ]
        return WorkerRecord(**dict(zip(keys, row)))

    def list_workers(self, status: str | None = None, batch_id: str | None = None) -> list[WorkerRecord]:
        sql = "SELECT worker_id FROM workers WHERE 1=1"
        args: list[Any] = []
        if status:
            sql += " AND status=?"
            args.append(status)
        if batch_id:
            sql += " AND batch_id=?"
            args.append(batch_id)
        sql += " ORDER BY created_ts"
        with self._lock:
            ids = [r[0] for r in self.conn.execute(sql, args).fetchall()]
        return [r for r in (self.load(i) for i in ids) if r is not None]

    def mark_interrupted(self) -> int:
        """启动时调用：上次进程挂掉时还在 running 的，标成 interrupted 等续跑。

        这是存算分离真正兑现价值的地方——不标的话它们会永远卡在 running，
        既不会被续跑，也不会被统计进去。
        """
        with self._lock:
            cur = self.conn.execute(
                "UPDATE workers SET status=?, updated_ts=? WHERE status=?",
                (INTERRUPTED, time.time(), RUNNING),
            )
            self.conn.commit()
            return cur.rowcount

    def count(self) -> int:
        with self._lock:
            return self.conn.execute("SELECT COUNT(*) FROM workers").fetchone()[0]

    def close(self) -> None:
        self.conn.close()
