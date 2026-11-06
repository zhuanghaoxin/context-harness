"""F4 回归：续跑的账目连续性与工具副作用幂等。

两个以前不成立的事实：
  1. resume 只恢复 pinned/history/step，meta 里的 usage/cost 从不读回 —— 续跑后的
     成本从 0 起算，"单任务成本"在续跑场景下直接失真。
  2. 工具副作用发生在"步末保存"之前。进程在一步中间挂掉，重跑会把同一批调用再执行一遍
     （写文件、发消息都不幂等）。现在按 (run_id, step, 调用序号:签名) 回放已执行结果。
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ch.agent import Agent
from ch.config import Config
from ch.tools import ToolRegistry, ToolSpec
from ch.types import LLMResponse, Message, ToolCall, Usage


class _ScriptedRouter:
    """按脚本产生响应；每个脚本项要么是工具调用列表，要么是最终文本。"""

    def __init__(self, script: list[dict]):
        self.script = script
        self.calls = 0

    def complete(self, messages, tools=None) -> LLMResponse:
        spec = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        usage = spec.get("usage", Usage(prompt_tokens=10, completion_tokens=5))
        cost = spec.get("cost", 0.01)
        if "text" in spec:
            return LLMResponse(
                message=Message(role="assistant", content=spec["text"]),
                usage=usage, model="fake", cost_usd=cost,
            )
        calls = [
            ToolCall(id=f"call_{self.calls}_{i}", name=name, arguments=args)
            for i, (name, args) in enumerate(spec["tools"])
        ]
        return LLMResponse(
            message=Message(role="assistant", content="", tool_calls=calls),
            usage=usage, model="fake", cost_usd=cost,
        )


class RunResumeTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.cfg = Config(context_budget=2000, db_path=str(self.dir / "h.sqlite3"),
                          artifact_dir=str(self.dir / "art"))
        self.cfg.ensure_dirs()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _registry(self, counter: list[int]) -> ToolRegistry:
        def count() -> str:
            counter[0] += 1
            return f"count={counter[0]}"

        def boom() -> str:
            # BaseException 不会被 ToolRegistry 的 `except Exception` 吞掉，
            # 用它精确模拟"一步中间进程挂掉"。
            raise KeyboardInterrupt("simulated crash mid-step")

        return (
            ToolRegistry(approval=lambda n, a: True)
            .register(ToolSpec("count", "计数一次", {"type": "object", "properties": {}}, count))
            .register(ToolSpec("boom", "模拟崩溃", {"type": "object", "properties": {}}, boom))
        )

    def test_resume_replays_tool_result_without_reexecuting(self):
        counter = [0]
        reg = self._registry(counter)
        crash = _ScriptedRouter([{"tools": [("count", {}), ("boom", {})]}])
        agent = Agent(crash, reg, self.cfg, checkpoint=None)
        with self.assertRaises(KeyboardInterrupt):
            agent.run("崩溃任务", run_id="r-crash")
        self.assertEqual(counter[0], 1)  # 崩溃前确实执行过一次

        resume = _ScriptedRouter([{"tools": [("count", {})]}, {"text": "done"}])
        agent2 = Agent(resume, reg, self.cfg, checkpoint=None)
        res = agent2.run("崩溃任务", run_id="r-crash", resume=True)
        agent2.close()

        self.assertEqual(res.stop_reason, "final")
        self.assertEqual(res.answer, "done")
        # 关键断言：续跑没有把 count 再调一次，用的是回放结果
        self.assertEqual(counter[0], 1)

    def test_resume_accumulates_usage_and_cost(self):
        reg = self._registry([0])
        first = _ScriptedRouter([{"text": "第一次", "usage": Usage(100, 50), "cost": 0.02}])
        agent = Agent(first, reg, self.cfg, checkpoint=None)
        res1 = agent.run("账目任务", run_id="r-usage")
        agent.close()
        self.assertEqual(res1.usage.total, 150)
        self.assertAlmostEqual(res1.cost_usd, 0.02)

        second = _ScriptedRouter([{"text": "第二次", "usage": Usage(30, 20), "cost": 0.01}])
        agent2 = Agent(second, reg, self.cfg, checkpoint=None)
        res2 = agent2.run("账目任务", run_id="r-usage", resume=True)
        agent2.close()

        self.assertEqual(res2.stop_reason, "final")
        self.assertEqual(res2.answer, "第二次")
        self.assertEqual(res2.usage.prompt_tokens, 130)
        self.assertEqual(res2.usage.completion_tokens, 70)
        self.assertAlmostEqual(res2.cost_usd, 0.03)


if __name__ == "__main__":
    unittest.main()
