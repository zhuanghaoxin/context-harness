"""S4 回归：LLM 重试（F6）、DAG 真续跑 / failed-skipped 落库（F8）、评测 CLI 可复现（T-5）。

这三件事的共同点：以前"看起来有、实际没有"。
  - F6：旧实现只捕 HTTPError，429/超时直接炸；空 choices 被 `[{}]` 兜底成空回答。
  - F8：DAG 调 Agent 时没传 resume=True，节点异常裸抛、不落库，"层级续跑"名不副实。
  - T-5：`ch.evaluate` 默认按 cwd 找 tasks.json，换目录就裸崩；输出文件名带时间戳无法 diff。
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]

from ch.checkpoint import CheckpointStore  # noqa: E402
from ch.config import Config  # noqa: E402
from ch.dag import DAG, DAGRunError, Node  # noqa: E402
from ch.llm import LLMError, OpenAICompatLLM  # noqa: E402
from ch.tokens import estimate  # noqa: E402
from ch.types import Message, RunResult, Usage  # noqa: E402


def _cfg() -> Config:
    return Config(base_url="http://127.0.0.1:9/v1", api_key="test-key", model="test-model")


class _Resp:
    """urlopen 的最小替身。"""

    def __init__(self, payload: dict):
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> "_Resp":
        return self

    def __exit__(self, *exc) -> bool:
        return False


def _ok_body(text: str = "ok") -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    }


def _http_error(code: int, msg: str = "err") -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x", code, msg, None, io.BytesIO(b'{"error":"boom"}'))


class LLMRetryTest(unittest.TestCase):
    """F6：429/5xx/网络错误退避重试；4xx 不重试；空 choices 显式报错。"""

    def test_retries_429_then_succeeds(self):
        calls: list[int] = []

        def fake(req, timeout=None):
            calls.append(1)
            if len(calls) == 1:
                raise _http_error(429, "Too Many Requests")
            return _Resp(_ok_body("recovered"))

        with mock.patch("ch.llm.urllib.request.urlopen", side_effect=fake):
            llm = OpenAICompatLLM(_cfg(), retries=2, backoff_s=0.0)
            resp = llm.complete([Message(role="user", content="hi")])
        self.assertEqual(resp.message.content, "recovered")
        self.assertEqual(len(calls), 2, "429 应当重试一次后成功")

    def test_client_error_is_not_retried(self):
        calls: list[int] = []

        def fake(req, timeout=None):
            calls.append(1)
            raise _http_error(401, "Unauthorized")

        with mock.patch("ch.llm.urllib.request.urlopen", side_effect=fake):
            llm = OpenAICompatLLM(_cfg(), retries=3, backoff_s=0.0)
            with self.assertRaises(LLMError) as ctx:
                llm.complete([Message(role="user", content="hi")])
        self.assertEqual(len(calls), 1, "401 重试只会浪费额度")
        self.assertFalse(ctx.exception.retryable)

    def test_server_error_exhausts_retries(self):
        calls: list[int] = []

        def fake(req, timeout=None):
            calls.append(1)
            raise _http_error(503, "Service Unavailable")

        with mock.patch("ch.llm.urllib.request.urlopen", side_effect=fake):
            llm = OpenAICompatLLM(_cfg(), retries=1, backoff_s=0.0)
            with self.assertRaises(LLMError):
                llm.complete([Message(role="user", content="hi")])
        self.assertEqual(len(calls), 2, "retries=1 表示总共尝试 2 次")

    def test_network_error_is_retried(self):
        calls: list[int] = []

        def fake(req, timeout=None):
            calls.append(1)
            raise urllib.error.URLError("connection reset")

        with mock.patch("ch.llm.urllib.request.urlopen", side_effect=fake):
            llm = OpenAICompatLLM(_cfg(), retries=2, backoff_s=0.0)
            with self.assertRaises(LLMError):
                llm.complete([Message(role="user", content="hi")])
        self.assertEqual(len(calls), 3)

    def test_empty_choices_raises_instead_of_empty_answer(self):
        with mock.patch("ch.llm.urllib.request.urlopen", return_value=_Resp({"choices": []})):
            llm = OpenAICompatLLM(_cfg(), retries=0)
            with self.assertRaises(LLMError) as ctx:
                llm.complete([Message(role="user", content="hi")])
        self.assertIn("空 choices", str(ctx.exception))

    def test_non_json_body_raises(self):
        class Raw(_Resp):
            def __init__(self):
                self._raw = b"<html>gateway</html>"

        with mock.patch("ch.llm.urllib.request.urlopen", return_value=Raw()):
            llm = OpenAICompatLLM(_cfg(), retries=0)
            with self.assertRaises(LLMError):
                llm.complete([Message(role="user", content="hi")])


class _FakeAgent:
    def __init__(self, log: list[dict], name: str, fail: bool = False, answer: str = "ok"):
        self.log = log
        self.name = name
        self.fail = fail
        self.answer = answer

    def run(self, task, run_id=None, resume=False):
        self.log.append({"node": self.name, "run_id": run_id, "resume": resume, "task": task})
        if self.fail:
            raise RuntimeError(f"{self.name} 炸了")
        return RunResult(answer=self.answer, stop_reason="final", steps=1, usage=Usage(), cost_usd=0.0)


class DAGResumeTest(unittest.TestCase):
    """F8：resume 透传、失败落库、下游 skipped、上游输出按 token 预算注入。"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.store = CheckpointStore(self.tmp / "cp.sqlite3")

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _dag(self) -> DAG:
        return (
            DAG(name="t-dag")
            .add(Node("a", "任务 A"))
            .add(Node("b", "任务 B", deps=("a",)))
            .add(Node("c", "任务 C", deps=("b",)))
        )

    def test_failed_node_is_persisted_and_downstream_skipped(self):
        log: list[dict] = []
        factory = lambda n: _FakeAgent(log, n.name, fail=(n.name == "b"))  # noqa: E731

        with self.assertRaises(DAGRunError) as ctx:
            self._dag().run(factory, dag_id="d1", checkpoint=self.store)

        err = ctx.exception
        self.assertIn("b", err.failures)
        self.assertEqual(err.skipped, ["c"])
        status = self.store.node_status("d1")
        self.assertEqual(status["a"][0], "completed")
        self.assertEqual(status["b"][0], "failed")
        self.assertIn("炸了", status["b"][1])
        self.assertEqual(status["c"][0], "skipped")
        self.assertEqual([e["node"] for e in log], ["a", "b"], "失败后不再执行下游")

    def test_resume_skips_completed_and_passes_resume_flag(self):
        log1: list[dict] = []
        factory1 = lambda n: _FakeAgent(log1, n.name, fail=(n.name == "b"))  # noqa: E731
        with self.assertRaises(DAGRunError):
            self._dag().run(factory1, dag_id="d2", checkpoint=self.store)

        # 第二次：a 已完成应被跳过；b 用 resume=True 从断点继续
        log2: list[dict] = []
        factory2 = lambda n: _FakeAgent(log2, n.name)  # noqa: E731
        out = self._dag().run(factory2, dag_id="d2", checkpoint=self.store, resume=True)

        self.assertEqual([e["node"] for e in log2], ["b", "c"], "completed 节点不应重跑")
        self.assertEqual(log2[0]["resume"], True, "resume 必须透传给 Agent")
        self.assertEqual(out["outputs"]["c"], "ok")
        self.assertEqual(out["skipped"], [])

    def test_full_rerun_after_success_executes_nothing(self):
        log1: list[dict] = []
        out1 = self._dag().run(lambda n: _FakeAgent(log1, n.name), dag_id="d3", checkpoint=self.store)
        self.assertEqual(set(out1["outputs"]), {"a", "b", "c"})

        log2: list[dict] = []
        out2 = self._dag().run(lambda n: _FakeAgent(log2, n.name), dag_id="d3",
                               checkpoint=self.store, resume=True)
        self.assertEqual(log2, [], "全部 completed 时续跑应零执行")
        self.assertEqual(out2["outputs"], out1["outputs"], "续跑输出应与首跑一致")

    def test_upstream_output_is_token_budgeted(self):
        log: list[dict] = []
        big = "汉" * 5000

        def factory(n):
            return _FakeAgent(log, n.name, answer=big if n.name == "a" else "ok")

        dag = DAG(name="t-big").add(Node("a", "A")).add(Node("b", "B", deps=("a",)))
        dag.run(factory, dag_id="d4", checkpoint=self.store)

        task_b = log[-1]["task"]
        self.assertIn("上游输出已按预算截断", task_b)
        self.assertNotIn(big, task_b)
        self.assertLessEqual(estimate(task_b), 800 + 64, "上游输出注入应受 token 预算约束")


class EvaluateCLITest(unittest.TestCase):
    """T-5：从任意 cwd 可跑；缺失任务集友好退出；--repeat 校验可复现。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _run(self, *args: str):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src")
        return subprocess.run(
            [sys.executable, "-m", "ch.evaluate", *args],
            cwd=str(self.tmp), env=env, capture_output=True, text=True, timeout=300,
        )

    def test_runs_from_foreign_cwd_with_repeat(self):
        # 用 full（记忆 + 压缩）而不是 budget：它能同时暴露 sqlite 句柄未释放的问题
        # （旧版第二次 repeat 删库时报 PermissionError: WinError 32）。
        out = self.tmp / "eval.json"
        proc = self._run("--variants", "full", "--repeat", "2", "--out", str(out))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(out.exists())
        data = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(data["repeat"], 2)
        self.assertEqual(len(data["results"]), 1)
        self.assertGreater(data["results"][0]["summary"]["pass_rate"], 0)

    def test_missing_tasks_file_exits_2_with_hint(self):
        proc = self._run("--tasks", str(self.tmp / "nope.json"))
        self.assertEqual(proc.returncode, 2)
        self.assertIn("找不到任务集", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr, "缺文件不应裸栈")


if __name__ == "__main__":
    unittest.main()
