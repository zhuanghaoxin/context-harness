"""Checkpoint：每步持久化，支持断点续跑。

要回答的问题："长任务跑到第 30 步挂了怎么办？"
答案不是"重跑一遍"（token 成本已经付过了），而是：
  - 每步把 (pinned messages, history) 序列化进 sqlite；
  - 恢复时按 run_id 载入，从 step+1 继续；
  - 已完成节点标记 completed，DAG 重跑时直接跳过。
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import Message


@dataclass
class Snapshot:
    run_id: str
    step: int
    task: str
    pinned: list[Message]
    history: list[Message]
    meta: dict[str, Any]
    updated_ts: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "step": self.step,
            "task": self.task,
            "pinned": [m.to_dict() for m in self.pinned],
            "history": [m.to_dict() for m in self.history],
            "meta": self.meta,
            "updated_ts": self.updated_ts,
        }


class CheckpointStore:
    """每步快照的 sqlite 落点。

    并发约束（多 Worker 同时落盘是常态，不是边角场景）：
      - 一条连接被多线程共用（check_same_thread=False），所以应用层用 RLock 串行化；
      - 跨连接/跨进程靠 WAL + busy_timeout，而不是"运气好不撞锁"；
      - synchronous=NORMAL 在 WAL 下仍然崩溃安全，但少了每次 fsync 的抖动。
    """

    def __init__(self, db_path: str | Path, timeout: float = 5.0):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._closed = False
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=timeout)
        with self._lock:
            # WAL：读写不互斥，多连接并发时不再直接 database is locked
            # busy_timeout：真撞上写锁时等一会儿，而不是立刻抛错
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute("PRAGMA busy_timeout=5000")
            self.conn.execute(
                """CREATE TABLE IF NOT EXISTS runs(
                       run_id TEXT PRIMARY KEY, step INTEGER, task TEXT, state TEXT, updated_ts REAL)"""
            )
            self.conn.execute(
                """CREATE TABLE IF NOT EXISTS nodes(
                       dag_id TEXT, node TEXT, status TEXT, output TEXT, updated_ts REAL,
                       PRIMARY KEY(dag_id, node))"""
            )
            self.conn.commit()

    def _conn(self) -> sqlite3.Connection:
        if self._closed or self.conn is None:
            raise RuntimeError("CheckpointStore 已关闭")
        return self.conn

    # ------------------------------------------------------------- runs
    def save(
        self,
        run_id: str,
        step: int,
        task: str,
        pinned: list[Message],
        history: list[Message],
        meta: dict[str, Any] | None = None,
    ) -> None:
        snap = Snapshot(run_id, step, task, pinned, history, meta or {}, time.time())
        with self._lock:
            conn = self._conn()
            conn.execute(
                "INSERT OR REPLACE INTO runs(run_id, step, task, state, updated_ts) VALUES (?,?,?,?,?)",
                (run_id, step, task, json.dumps(snap.to_dict(), ensure_ascii=False), snap.updated_ts),
            )
            conn.commit()

    def load(self, run_id: str) -> Snapshot | None:
        with self._lock:
            row = self._conn().execute("SELECT state FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if not row:
            return None
        d = json.loads(row[0])
        return Snapshot(
            run_id=d["run_id"],
            step=d["step"],
            task=d["task"],
            pinned=[Message.from_dict(m) for m in d["pinned"]],
            history=[Message.from_dict(m) for m in d["history"]],
            meta=d.get("meta") or {},
            updated_ts=d.get("updated_ts") or 0.0,
        )

    def list_runs(self, limit: int = 20) -> list[tuple[str, int, str, float]]:
        with self._lock:
            return self._conn().execute(
                "SELECT run_id, step, task, updated_ts FROM runs ORDER BY updated_ts DESC LIMIT ?", (limit,)
            ).fetchall()

    # ------------------------------------------------------------- dag nodes
    def mark_node(self, dag_id: str, node: str, status: str, output: str = "") -> None:
        with self._lock:
            conn = self._conn()
            conn.execute(
                "INSERT OR REPLACE INTO nodes(dag_id, node, status, output, updated_ts) VALUES (?,?,?,?,?)",
                (dag_id, node, status, output, time.time()),
            )
            conn.commit()

    def close(self) -> None:
        """幂等关闭：WorkerPool/Agent 的生命周期终点会调，异常路径也可能调。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self.conn is not None:
                try:
                    self.conn.close()
                finally:
                    self.conn = None

    @property
    def closed(self) -> bool:
        return self._closed

    def node_status(self, dag_id: str) -> dict[str, tuple[str, str]]:
        with self._lock:
            rows = self._conn().execute(
                "SELECT node, status, output FROM nodes WHERE dag_id=?", (dag_id,)
            ).fetchall()
        return {r[0]: (r[1], r[2]) for r in rows}
