"""统一日志入口：只用标准库 logging，零依赖。

之前全仓没有一条日志：在线跑出问题时只能靠 trace 事后回放，进程里发生了什么
（重试了几次、Worker 复用了谁、任务停在哪）完全看不到。这里给一个统一入口，
级别由环境变量 CH_LOG_LEVEL 控制（默认 WARNING，不打扰正常输出）。
"""
from __future__ import annotations

import logging
import os

FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_configured = False


def configure(level: str | int | None = None) -> None:
    """幂等配置：重复调用只更新级别，不会重复挂 handler。"""
    global _configured
    logger = logging.getLogger("ch")
    if not _configured:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(FORMAT))
        logger.addHandler(handler)
        logger.propagate = False  # 避免被 root handler 打印两遍
        _configured = True
    value = level if level is not None else os.environ.get("CH_LOG_LEVEL", "WARNING")
    if isinstance(value, str):
        value = value.strip().upper() or "WARNING"
    logger.setLevel(value)


def get_logger(name: str) -> logging.Logger:
    configure()
    return logging.getLogger(f"ch.{name}")
