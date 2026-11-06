"""向量化层：本地哈希 embedding（零依赖） + 真实 embedding 服务（OpenAI 兼容）。

为什么要做成可插拔：本地哈希只能抓字面 2-gram 重合，"同义改写"（"预算不够用"
vs "上下文窗口快满了"）几乎认不出来。真实的 Router 话题判定必须用语义向量，
但 CI 和离线演示又不能依赖网络和 key，所以两套实现必须能换。

三个实现，用途不同：

    HashingEmbedder  : 零依赖、确定性、离线可复现，用于单测和演示
    RemoteEmbedder   : 调 OpenAI 兼容的 /embeddings，带 sqlite 缓存（同一句只花钱一次）
    ResilientEmbedder: 包一层熔断——远端挂了就整场降级到本地，不让主流程崩

一个必须注意的坑：两种实现维度不同（1536 vs 512），一旦中途降级，
库里会同时存在两种维度的向量。余弦相似度对维度不等长会直接算错，
所以 EpisodicStore 检索时必须按维度过滤（见 memory.py 的 search）。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Any, Protocol


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]:
        ...


class EmbeddingUnavailable(RuntimeError):
    """远端 embedding 不可用（网络/鉴权/限流）。调用方决定降级还是失败。"""


class HashingEmbedder:
    """零依赖的确定性向量化：字符 n-gram + ascii 词，哈希到固定维度后 L2 归一化。

    对中文按 2-gram 切分（"内存管理" -> 内存/存管/管理），对英文按词切分。
    召回质量不如真 embedding，但对"同字面"足够，且完全离线可复现。
    """

    name = "hashing"

    def __init__(self, dim: int = 512, ngram: int = 2):
        self.dim = dim
        self.ngram = ngram

    def _tokens(self, text: str) -> list[str]:
        text = text.lower()
        toks: list[str] = []
        for w in re.findall(r"[a-z0-9_]+", text):
            toks.append(w)
        cjk = re.sub(r"[^\u3400-\u9fff]", "", text)
        for i in range(0, max(0, len(cjk) - self.ngram + 1)):
            toks.append(cjk[i : i + self.ngram])
        return toks

    def embed(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in self._tokens(text):
            h = zlib.crc32(tok.encode("utf-8"))
            idx = h % self.dim
            sign = 1.0 if (h >> 16) & 1 else -1.0
            v[idx] += sign
        return _l2(v)


def _l2(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def similarity(a: list[float], b: list[float]) -> float:
    return cosine(a, b)


# ---------------------------------------------------------------- 远端服务


class EmbeddingCache:
    """按 (model, sha256(text)) 缓存向量。同一句话只付一次钱。"""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = str(db_path) if db_path else ":memory:"
        if db_path:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS embedding_cache(
                   model TEXT, sha TEXT, dim INTEGER, vec TEXT, ts REAL,
                   PRIMARY KEY(model, sha))"""
        )
        self.conn.commit()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _sha(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, model: str, text: str) -> list[float] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT vec FROM embedding_cache WHERE model=? AND sha=?", (model, self._sha(text))
            ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(row[0])

    def put(self, model: str, text: str, vec: list[float]) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO embedding_cache(model, sha, dim, vec, ts) VALUES (?,?,?,?,?)",
                (model, self._sha(text), len(vec), json.dumps(vec), time.time()),
            )
            self.conn.commit()

    def __len__(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0]

    def close(self) -> None:
        self.conn.close()


class RemoteEmbedder:
    """OpenAI 兼容的 /embeddings 客户端。零依赖（urllib），带缓存与重试。"""

    name = "remote"

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        model: str = "text-embedding-3-small",
        dim: int = 0,
        timeout: float = 30.0,
        cache: EmbeddingCache | None = None,
        retries: int = 1,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dim = dim  # 拿到第一条响应后自动填
        self.timeout = timeout
        self.cache = cache
        self.retries = retries
        self.calls = 0

    def _post(self, inputs: list[str]) -> list[list[float]]:
        body = json.dumps({"model": self.model, "input": inputs}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/embeddings", data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:  # 保留服务端原文，否则排查全靠猜
            detail = e.read().decode("utf-8", errors="replace")[:400]
            raise EmbeddingUnavailable(f"HTTP {e.code}: {detail}") from e
        except Exception as e:
            raise EmbeddingUnavailable(f"{type(e).__name__}: {e}") from e

        items = data.get("data") or []
        if len(items) != len(inputs):
            raise EmbeddingUnavailable(f"返回条数不符: 期望 {len(inputs)} 实得 {len(items)}")
        vecs = [item["embedding"] for item in items]
        if vecs and not self.dim:
            self.dim = len(vecs[0])
        return [_l2(v) for v in vecs]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float] | None] = [None] * len(texts)
        missing: list[tuple[int, str]] = []
        cache = self.cache  # 注意：不能用 `if self.cache`，EmbeddingCache 定义了 __len__，
        # 空缓存会被判成 False，导致永远不读也不写缓存（这个坑被单测抓出来了）。
        for i, t in enumerate(texts):
            cached = cache.get(self.model, t) if cache is not None else None
            if cached is not None:
                out[i] = cached
            else:
                missing.append((i, t))
        if not missing:
            return [v for v in out if v is not None]

        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                vecs = self._post([t for _, t in missing])
                self.calls += 1
                for (i, t), v in zip(missing, vecs):
                    out[i] = v
                    if cache is not None:
                        cache.put(self.model, t, v)
                return [v for v in out if v is not None]
            except EmbeddingUnavailable as e:
                last = e
                if attempt < self.retries:
                    time.sleep(0.2 * (attempt + 1))
        raise last or EmbeddingUnavailable("unknown")

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]


class ResilientEmbedder:
    """远端优先，失败后整场熔断降级到本地实现。

    为什么是"整场"降级而不是逐次重试：逐次重试会让同一批文档里一半是
    1536 维、一半是 512 维，相似度计算直接失效。降级后只用一个维度，
    代价是召回变差，但至少可比。
    """

    def __init__(self, primary: Any, fallback: HashingEmbedder | None = None):
        self.primary = primary
        self.fallback = fallback or HashingEmbedder()
        self.degraded = False
        self.failures = 0
        self.last_error = ""

    @property
    def dim(self) -> int:
        if not self.degraded and getattr(self.primary, "dim", 0):
            return int(self.primary.dim)
        return self.fallback.dim

    @property
    def name(self) -> str:
        if self.degraded:
            return f"{getattr(self.primary, 'name', 'remote')}->degraded:{self.fallback.name}"
        return getattr(self.primary, "name", "remote")

    def embed(self, text: str) -> list[float]:
        if not self.degraded:
            try:
                return self.primary.embed(text)
            except EmbeddingUnavailable as e:
                self.degraded = True
                self.failures += 1
                self.last_error = str(e)
        return self.fallback.embed(text)


def make_embedder(cfg: Any) -> Any:
    """按配置挑实现：配了 embedding_model 且非离线 -> 远端 + 熔断；否则本地哈希。"""
    model = getattr(cfg, "embedding_model", "") or ""
    if not model:
        return HashingEmbedder(dim=getattr(cfg, "embedding_dim", 512))
    if getattr(cfg, "offline", False):
        return HashingEmbedder(dim=getattr(cfg, "embedding_dim", 512))
    cache = EmbeddingCache(getattr(cfg, "embedding_cache_path", "") or None)
    # embedding 允许独立端点：DeepSeek 这类只做 chat 的厂商没有 /embeddings
    base = getattr(cfg, "embedding_base_url", "") or getattr(cfg, "base_url", "")
    key = getattr(cfg, "embedding_api_key", "") or getattr(cfg, "api_key", "")
    primary = RemoteEmbedder(
        base_url=base,
        api_key=key,
        model=model,
        timeout=getattr(cfg, "embedding_timeout", 30.0),
        cache=cache,
    )
    return ResilientEmbedder(primary, HashingEmbedder(dim=getattr(cfg, "embedding_dim", 512)))
