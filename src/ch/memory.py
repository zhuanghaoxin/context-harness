"""三层记忆。

要回答的问题："Memory 怎么实现？Mem0 和 Zep 的区别在哪？"
本实现给出可解释的三层划分，并且每一层的写入策略都是显式的：

  工作记忆 WorkingMemory
      = 当前 context window，容量 = token 预算，由 budget.py 裁剪。生命周期一次 run。

  情景记忆 EpisodicStore
      = "发生过什么"，按时间追加，向量检索 top-k。落地在 sqlite，
        默认用 HashingEmbedder（零依赖的 bag-of-words 哈希向量），
        可替换成任意 embedding API——接口只有 embed(text)->vector。

  语义记忆 SemanticMemory
      = "事实是什么"，带 key 的结构化事实，写入时做 **去重 / 冲突更新 / 时间衰减**：
        - 同 key 同 value  → reinforce（confidence 上升，更新时间）
        - 同 key 不同 value → supersede（旧事实置 valid_until，新事实接管）
        - 检索打分 = 字面相关度 × confidence × 0.5^(age/half_life_days)

为什么不用 mem0：这里要的是"每一步可解释"，且需要把去重/冲突/衰减三条策略
显式写出来作为设计说明；用 mem0 就变成调库了。
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import time
import uuid
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

from .embedding import Embedder, HashingEmbedder, cosine, make_embedder  # noqa: F401 - 兼容旧的导入路径
from .tokens import estimate

# --------------------------------------------------------------- embedder
# 实现搬到了 embedding.py：本地哈希 / 远端 OpenAI 兼容 / 带熔断的组合。
# 这里保留同名再导出，老代码 from .memory import HashingEmbedder 仍然有效。


# --------------------------------------------------------------- working memory


@dataclass
class WorkingMemory:
    """当前 run 的上下文窗口。真正的裁剪发生在 budget.TokenBudget。"""

    max_tokens: int = 8000
    messages: list[Any] = field(default_factory=list)

    def push(self, msg: Any) -> None:
        self.messages.append(msg)

    def snapshot(self) -> list[Any]:
        return list(self.messages)

    def clear(self) -> None:
        self.messages.clear()


# --------------------------------------------------------------- episodic


@dataclass
class Episode:
    id: str
    ts: float
    text: str
    meta: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


class EpisodicStore:
    def __init__(self, db_path: str | Path, embedder: Embedder | None = None, namespace: str = "default"):
        self.embedder = embedder or HashingEmbedder()
        self.namespace = namespace
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS episodes(
                   id TEXT PRIMARY KEY,
                   ns TEXT, ts REAL, text TEXT, embedding TEXT, meta TEXT)"""
        )
        self.conn.commit()

    def add(self, text: str, meta: dict[str, Any] | None = None) -> str:
        eid = f"ep_{int(time.time()*1000)}_{zlib.crc32(text.encode('utf-8')) % 100000}"
        emb = self.embedder.embed(text)
        self.conn.execute(
            "INSERT OR REPLACE INTO episodes(id, ns, ts, text, embedding, meta) VALUES (?,?,?,?,?,?)",
            (eid, self.namespace, time.time(), text, json.dumps(emb), json.dumps(meta or {}, ensure_ascii=False)),
        )
        self.conn.commit()
        return eid

    def search(self, query: str, k: int = 5, min_score: float = 0.05) -> list[Episode]:
        qv = self.embedder.embed(query)
        rows = self.conn.execute(
            "SELECT id, ts, text, embedding, meta FROM episodes WHERE ns=?", (self.namespace,)
        ).fetchall()
        scored: list[Episode] = []
        skipped_dim = 0
        for eid, ts, text, emb, meta in rows:
            vec = json.loads(emb)
            # 中途降级（远端挂了换本地）会让库里同时存在两种维度的向量。
            # 维度不同的向量做点积会静默算错，必须跳过而不是硬算。
            if len(vec) != len(qv):
                skipped_dim += 1
                continue
            s = cosine(qv, vec)
            if s >= min_score:
                scored.append(Episode(eid, ts, text, json.loads(meta or "{}"), s))
        self.last_skipped_dim = skipped_dim
        scored.sort(key=lambda e: -e.score)
        return scored[:k]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM episodes WHERE ns=?", (self.namespace,)).fetchone()[0]

    def close(self) -> None:
        self.conn.close()


# --------------------------------------------------------------- semantic


@dataclass
class Fact:
    id: str
    key: str
    value: str
    source: str
    ts: float
    updated_ts: float
    confidence: float
    hits: int
    valid_until: float | None = None
    score: float = 0.0


class SemanticMemory:
    """带去重 / 冲突更新 / 时间衰减的结构化事实库。"""

    def __init__(self, db_path: str | Path, half_life_days: float = 14.0, namespace: str = "default"):
        self.half_life_days = half_life_days
        self.namespace = namespace
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS facts(
                   id TEXT PRIMARY KEY, ns TEXT, key TEXT, value TEXT, source TEXT,
                   ts REAL, updated_ts REAL, confidence REAL, hits INTEGER,
                   valid_until REAL, superseded_by TEXT)"""
        )
        self.conn.commit()

    @staticmethod
    def _norm_key(k: str) -> str:
        return re.sub(r"\s+", " ", k.strip().lower())

    @staticmethod
    def _norm_val(v: str) -> str:
        return re.sub(r"\s+", " ", str(v).strip())

    def remember(self, key: str, value: str, source: str = "agent", confidence: float = 0.7) -> str:
        """返回 'inserted' / 'reinforced' / 'superseded'，三种策略显式可见。"""
        nk, nv = self._norm_key(key), self._norm_val(value)
        now = time.time()
        row = self.conn.execute(
            "SELECT id, value, confidence, hits FROM facts WHERE ns=? AND key=? AND valid_until IS NULL",
            (self.namespace, nk),
        ).fetchone()

        if row is None:
            fid = f"f_{uuid.uuid4().hex[:12]}"
            self.conn.execute(
                """INSERT INTO facts(id, ns, key, value, source, ts, updated_ts, confidence, hits, valid_until, superseded_by)
                   VALUES (?,?,?,?,?,?,?,?,0,NULL,NULL)""",
                (fid, self.namespace, nk, nv, source, now, now, confidence, ),
            )
            self.conn.commit()
            return "inserted"

        fid, old_val, conf, hits = row
        if self._norm_val(old_val) == nv:
            # 去重：同一事实重复出现 → 提升置信度，而不是新增一条
            self.conn.execute(
                "UPDATE facts SET hits=?, confidence=?, updated_ts=? WHERE id=?",
                (hits + 1, min(1.0, conf + 0.1), now, fid),
            )
            self.conn.commit()
            return "reinforced"

        # 冲突：旧事实封存，新事实接管（保留历史，可回溯）
        new_id = f"f_{uuid.uuid4().hex[:12]}"
        self.conn.execute("UPDATE facts SET valid_until=?, superseded_by=? WHERE id=?", (now, new_id, fid))
        self.conn.execute(
            """INSERT INTO facts(id, ns, key, value, source, ts, updated_ts, confidence, hits, valid_until, superseded_by)
               VALUES (?,?,?,?,?,?,?,?,0,NULL,NULL)""",
            (new_id, self.namespace, nk, nv, source, now, now, confidence),
        )
        self.conn.commit()
        return "superseded"

    def all_valid(self) -> list[Fact]:
        rows = self.conn.execute(
            "SELECT id,key,value,source,ts,updated_ts,confidence,hits FROM facts WHERE ns=? AND valid_until IS NULL",
            (self.namespace,),
        ).fetchall()
        return [Fact(*r) for r in rows]

    def close(self) -> None:
        self.conn.close()

    def recall(self, query: str, k: int = 8, now: float | None = None) -> list[Fact]:
        now = now or time.time()
        q = set(re.findall(r"[\w\u4e00-\u9fff]+", query.lower()))
        out: list[Fact] = []
        for f in self.all_valid():
            kw = set(re.findall(r"[\w\u4e00-\u9fff]+", (f.key + " " + f.value).lower()))
            overlap = len(q & kw) / max(1, len(q))
            age_days = max(0.0, (now - f.updated_ts) / 86400)
            decay = 0.5 ** (age_days / self.half_life_days)
            f.score = (0.35 + 0.65 * overlap) * f.confidence * decay
            out.append(f)
        out.sort(key=lambda f: -f.score)
        return out[:k]


# --------------------------------------------------------------- manager


class MemoryManager:
    """把三层记忆组装成一个"可注入上下文"的块，并受 token 预算约束。"""

    def __init__(
        self,
        db_path: str | Path,
        context_budget: int = 8000,
        half_life_days: float = 14.0,
        embedder: Embedder | None = None,
    ):
        self.db_path = str(db_path)
        self.working = WorkingMemory(max_tokens=context_budget)
        self.episodic = EpisodicStore(self.db_path, embedder=embedder)
        self.semantic = SemanticMemory(self.db_path, half_life_days=half_life_days)
        self.last_stats: dict[str, Any] = {}

    # 写入
    def record_episode(self, text: str, meta: dict[str, Any] | None = None) -> str:
        return self.episodic.add(text, meta)

    def remember_fact(self, key: str, value: str, source: str = "agent", confidence: float = 0.7) -> str:
        return self.semantic.remember(key, value, source, confidence)

    # 读取
    def recall_block(self, query: str, budget_tokens: int = 800, k_facts: int = 8, k_eps: int = 4) -> str:
        facts = self.semantic.recall(query, k=k_facts)
        eps = self.episodic.search(query, k=k_eps)

        lines: list[str] = []
        if facts:
            lines.append("【已知事实】")
            for f in facts:
                lines.append(f"- {f.key}: {f.value} (来源={f.source}, 置信={f.confidence:.2f}, 权重={f.score:.2f})")
        if eps:
            lines.append("【相关历史片段】")
            for e in eps:
                snippet = e.text.replace("\n", " ")[:160]
                lines.append(f"- ({time.strftime('%m-%d', time.localtime(e.ts))}) {snippet} [sim={e.score:.2f}]")

        block = "\n".join(lines)
        if not block:
            block = ""

        truncated = False
        if estimate(block) > budget_tokens:
            from .budget import truncate_to_tokens

            block = truncate_to_tokens(block, budget_tokens, marker="\n...[memory truncated by budget]")
            truncated = True

        self.last_stats = {
            "facts_returned": len(facts),
            "episodes_returned": len(eps),
            "memory_tokens": estimate(block),
            "truncated": truncated,
            "episodic_total": self.episodic.count(),
        }
        return block

    def stats(self) -> dict[str, Any]:
        return {
            "episodes": self.episodic.count(),
            "facts_valid": len(self.semantic.all_valid()),
            "last_recall": self.last_stats,
        }

    def close(self) -> None:
        self.episodic.close()
        self.semantic.close()
