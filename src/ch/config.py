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


class ConfigError(ValueError):
    """配置非法。带上变量名和原值，别让 int() 的报错裸奔到用户面前。"""


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        raise ConfigError(f"{name}={raw!r} 不是合法整数") from None
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name}={value} 必须 >= {minimum}")
    return value


def _env_float(name: str, default: float, lo: float | None = None, hi: float | None = None) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw.strip())
    except ValueError:
        raise ConfigError(f"{name}={raw!r} 不是合法浮点数") from None
    if lo is not None and value < lo:
        raise ConfigError(f"{name}={value} 必须 >= {lo}")
    if hi is not None and value > hi:
        raise ConfigError(f"{name}={value} 必须 <= {hi}")
    return value


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
        cfg = Config(
            base_url=g("CH_BASE_URL", ""),
            api_key=g("CH_API_KEY", ""),
            model=g("CH_MODEL", ""),
            small_model=g("CH_SMALL_MODEL", "") or g("CH_MODEL", ""),
            max_steps=_env_int("CH_MAX_STEPS", 16, minimum=1),
            context_budget=_env_int("CH_CONTEXT_BUDGET", 8000, minimum=256),
            db_path=g("CH_DB_PATH", ".artifacts/harness.sqlite3"),
            artifact_dir=g("CH_ARTIFACT_DIR", ".artifacts"),
            filter_enabled=_bool(g("CH_FILTER_ENABLED", "1")),
            filter_high=_env_float("CH_FILTER_HIGH", 0.70, 0.0, 1.0),
            filter_low=_env_float("CH_FILTER_LOW", 0.30, 0.0, 1.0),
            max_workers=_env_int("CH_MAX_WORKERS", 4, minimum=1),
            worker_reuse_threshold=_env_float("CH_WORKER_REUSE", 0.35, 0.0, 1.0),
            watchdog_stall_steps=_env_int("CH_WATCHDOG_STALL", 3, minimum=1),
            watchdog_topic_floor=_env_float("CH_WATCHDOG_TOPIC", 0.15, 0.0, 1.0),
            watchdog_min_text=_env_int("CH_WATCHDOG_MIN_TEXT", 40, minimum=1),
            sim_latency_s=_env_float("CH_SIM_LATENCY_MS", 0.0, 0.0) / 1000.0,
            embedding_model=g("CH_EMBEDDING_MODEL", ""),
            embedding_dim=_env_int("CH_EMBEDDING_DIM", 512, minimum=1),
            embedding_base_url=g("CH_EMBEDDING_BASE_URL", ""),
            embedding_api_key=g("CH_EMBEDDING_API_KEY", ""),
            embedding_cache_path=g("CH_EMBEDDING_CACHE", ".artifacts/embedding_cache.sqlite3"),
            embedding_timeout=_env_float("CH_EMBEDDING_TIMEOUT", 30.0, 0.1),
        )
        return cfg.validate()

    def validate(self) -> "Config":
        """字段级校验：一次性报出所有非法项，而不是撞到哪个崩哪个。"""
        problems: list[str] = []
        if self.max_steps < 1:
            problems.append(f"max_steps={self.max_steps} 必须 >= 1")
        if self.context_budget < 256:
            problems.append(f"context_budget={self.context_budget} 太小（>= 256）")
        if self.max_workers < 1:
            problems.append(f"max_workers={self.max_workers} 必须 >= 1")
        if self.queue_factor < 1:
            problems.append(f"queue_factor={self.queue_factor} 必须 >= 1")
        total = self.tool_share + self.history_share + self.memory_share
        if not 0.99 <= total <= 1.01:
            problems.append(f"预算比例之和 {total:.3f} 应约等于 1.0")
        if not 0.0 <= self.filter_low < self.filter_high <= 1.0:
            problems.append(
                f"filter_low={self.filter_low} 必须 < filter_high={self.filter_high}（且都在 [0,1]）"
            )
        if not 0.0 <= self.worker_reuse_threshold <= 1.0:
            problems.append(f"worker_reuse_threshold={self.worker_reuse_threshold} 必须在 [0,1]")
        if self.embedding_dim < 1:
            problems.append(f"embedding_dim={self.embedding_dim} 必须 >= 1")
        if self.embedding_timeout <= 0:
            problems.append(f"embedding_timeout={self.embedding_timeout} 必须 > 0")
        if problems:
            raise ConfigError("配置校验失败：" + "；".join(problems))
        return self

    def ensure_dirs(self) -> None:
        Path(self.artifact_dir).mkdir(parents=True, exist_ok=True)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def price(self, model: str) -> tuple[float, float]:
        return self.price_per_mtok.get(model, (0.0, 0.0))

    def cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        pin, pout = self.price(model)
        return (prompt_tokens * pin + completion_tokens * pout) / 1_000_000
