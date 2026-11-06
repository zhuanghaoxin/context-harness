"""配置：全部来自环境变量，可被 .env 文件覆盖（零依赖的极简 .env 解析）。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def _bool(v: str) -> bool:
    return v.strip().lower() not in ("", "0", "false", "no", "off")


@dataclass
class Config:
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    small_model: str = ""
    max_steps: int = 16
    context_budget: int = 8000
    db_path: str = ".artifacts/harness.sqlite3"
    artifact_dir: str = ".artifacts"
    # 预算分配比例（剩余预算的占比）：工具输出 / 历史对话 / 记忆召回
    tool_share: float = 0.45
    history_share: float = 0.35
    memory_share: float = 0.20
    # 目标重注入间隔（步）
    goal_reinject_every: int = 4
    # 同一工具+参数重复调用多少次判定为死循环
    repeat_limit: int = 3
    # ---- Super-agent：两级消息过滤
    filter_enabled: bool = True
    # 规则分高于 high 直接放行、低于 low 直接丢弃，只有灰区才调小模型（成本控制）
    filter_high: float = 0.70
    filter_low: float = 0.30
    # ---- Router-Worker 双层
    max_workers: int = 4
    # 待处理消息最多堆到 max_workers 的多少倍，超过才拒收（背压）
    queue_factor: int = 3
    # 新消息与某个活跃 worker 的任务相似度达到此值就复用该 worker（消息直达，不过路由）
    worker_reuse_threshold: float = 0.35
    # ---- Watchdog
    watchdog_stall_steps: int = 3
    watchdog_topic_floor: float = 0.15
    watchdog_min_text: int = 40
    # ---- 向量化：留空走本地哈希（零依赖），填了就走 OpenAI 兼容 /embeddings
    embedding_model: str = ""
    embedding_dim: int = 512
    # embedding 可以和 chat 用不同厂商（DeepSeek 没有 /embeddings），留空则复用 base_url/api_key
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_cache_path: str = ".artifacts/embedding_cache.sqlite3"
    embedding_timeout: float = 30.0
    # 单步模拟延迟（秒）：只用于离线复现 P95 端到端实验，在线时保持 0
    sim_latency_s: float = 0.0
    # 单价（USD / 1M tokens），用于 trace 里的成本统计
    price_per_mtok: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "deepseek-chat": (0.27, 1.10),
            "gpt-4o-mini": (0.15, 0.60),
            "gpt-4o": (2.50, 10.00),
        }
    )

    @property
    def offline(self) -> bool:
        return not (self.base_url and self.api_key and self.model)

    @staticmethod
    def from_env() -> "Config":
        load_dotenv()
        g = os.environ.get
        return Config(
            base_url=g("CH_BASE_URL", ""),
            api_key=g("CH_API_KEY", ""),
            model=g("CH_MODEL", ""),
            small_model=g("CH_SMALL_MODEL", "") or g("CH_MODEL", ""),
            max_steps=int(g("CH_MAX_STEPS", "16")),
            context_budget=int(g("CH_CONTEXT_BUDGET", "8000")),
            db_path=g("CH_DB_PATH", ".artifacts/harness.sqlite3"),
            artifact_dir=g("CH_ARTIFACT_DIR", ".artifacts"),
            filter_enabled=_bool(g("CH_FILTER_ENABLED", "1")),
            filter_high=float(g("CH_FILTER_HIGH", "0.70")),
            filter_low=float(g("CH_FILTER_LOW", "0.30")),
            max_workers=int(g("CH_MAX_WORKERS", "4")),
            worker_reuse_threshold=float(g("CH_WORKER_REUSE", "0.35")),
            watchdog_stall_steps=int(g("CH_WATCHDOG_STALL", "3")),
            watchdog_topic_floor=float(g("CH_WATCHDOG_TOPIC", "0.15")),
            watchdog_min_text=int(g("CH_WATCHDOG_MIN_TEXT", "40")),
            sim_latency_s=float(g("CH_SIM_LATENCY_MS", "0")) / 1000.0,
            embedding_model=g("CH_EMBEDDING_MODEL", ""),
            embedding_dim=int(g("CH_EMBEDDING_DIM", "512")),
            embedding_base_url=g("CH_EMBEDDING_BASE_URL", ""),
            embedding_api_key=g("CH_EMBEDDING_API_KEY", ""),
            embedding_cache_path=g("CH_EMBEDDING_CACHE", ".artifacts/embedding_cache.sqlite3"),
            embedding_timeout=float(g("CH_EMBEDDING_TIMEOUT", "30")),
        )

    def ensure_dirs(self) -> None:
        Path(self.artifact_dir).mkdir(parents=True, exist_ok=True)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def price(self, model: str) -> tuple[float, float]:
        return self.price_per_mtok.get(model, (0.0, 0.0))

    def cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        pin, pout = self.price(model)
        return (prompt_tokens * pin + completion_tokens * pout) / 1_000_000
