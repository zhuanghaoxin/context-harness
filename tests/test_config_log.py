"""F11 回归：配置字段级校验 + 运行期日志。

以前的两类问题：
  1. from_env 直接 int()/float()，一个手抖的环境变量就抛出裸 ValueError
     （"invalid literal for int()"），既不知道是哪个变量，也不会一次报全。
  2. 全仓零日志：重试了几次、任务停在哪、花了多少 token，进程里完全不可观测。
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ch.agent import Agent
from ch.config import Config, ConfigError
from ch.log import configure, get_logger
from ch.tools import ToolRegistry
from ch.types import LLMResponse, Message, Usage


class _EchoRouter:
    def complete(self, messages, tools=None) -> LLMResponse:
        return LLMResponse(
            message=Message(role="assistant", content="ok"),
            usage=Usage(prompt_tokens=7, completion_tokens=3),
            model="fake", cost_usd=0.001,
        )


class ConfigValidationTest(unittest.TestCase):
    def test_non_integer_env_names_the_variable(self):
        with mock.patch.dict(os.environ, {"CH_MAX_STEPS": "abc"}, clear=False):
            with self.assertRaises(ConfigError) as ctx:
                Config.from_env()
        msg = str(ctx.exception)
        self.assertIn("CH_MAX_STEPS", msg)
        self.assertIn("abc", msg)

    def test_out_of_range_env_is_rejected(self):
        with mock.patch.dict(os.environ, {"CH_CONTEXT_BUDGET": "100"}, clear=False):
            with self.assertRaises(ConfigError) as ctx:
                Config.from_env()
        self.assertIn("CH_CONTEXT_BUDGET", str(ctx.exception))

    def test_bad_float_env(self):
        with mock.patch.dict(os.environ, {"CH_FILTER_HIGH": "high"}, clear=False):
            with self.assertRaises(ConfigError) as ctx:
                Config.from_env()
        self.assertIn("CH_FILTER_HIGH", str(ctx.exception))

    def test_valid_env_round_trips(self):
        with mock.patch.dict(
            os.environ,
            {"CH_MAX_STEPS": "5", "CH_CONTEXT_BUDGET": "4096", "CH_FILTER_HIGH": "0.9"},
            clear=False,
        ):
            cfg = Config.from_env()
        self.assertEqual(cfg.max_steps, 5)
        self.assertEqual(cfg.context_budget, 4096)
        self.assertAlmostEqual(cfg.filter_high, 0.9)

    def test_validate_reports_all_problems_at_once(self):
        cfg = Config(
            max_steps=0,
            context_budget=10,
            max_workers=0,
            tool_share=0.9, history_share=0.9, memory_share=0.9,
            filter_low=0.8, filter_high=0.2,
        )
        with self.assertRaises(ConfigError) as ctx:
            cfg.validate()
        msg = str(ctx.exception)
        for needle in ("max_steps", "context_budget", "max_workers", "预算比例之和", "filter_low"):
            self.assertIn(needle, msg)

    def test_defaults_validate_clean(self):
        self.assertIs(Config().validate().__class__, Config)


class LoggingTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.cfg = Config(context_budget=1000, db_path=str(self.dir / "h.sqlite3"),
                          artifact_dir=str(self.dir / "art"))
        self.cfg.ensure_dirs()

    def tearDown(self):
        configure("WARNING")  # 别把 INFO 级别漏给后续测试，污染输出
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_get_logger_is_namespaced_and_idempotent(self):
        logger = get_logger("agent")
        self.assertEqual(logger.name, "ch.agent")
        configure("INFO")
        configure("INFO")
        root = logging.getLogger("ch")
        self.assertEqual(len(root.handlers), 1)

    def test_agent_emits_start_and_stop_records(self):
        agent = Agent(_EchoRouter(), ToolRegistry(approval=lambda n, a: True), self.cfg, checkpoint=None)
        with self.assertLogs("ch.agent", level="INFO") as captured:
            agent.run("日志任务", run_id="r-log")
        agent.close()
        text = "\n".join(captured.output)
        self.assertIn("r-log", text)
        self.assertIn("start resume=False", text)
        self.assertIn("stop=final", text)
        self.assertIn("prompt_tokens=7", text)


if __name__ == "__main__":
    unittest.main()
