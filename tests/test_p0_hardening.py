"""P0 硬化回归测试：预算配额、存档 ref 逃逸、沙箱前缀、checkpoint 并发。

这些用例对应审计里实测过的缺陷，修完后必须一直绿：
  - F1 中文按固定字符比例裁剪 → 实测预算 100 token 裁出 273 token；
  - F2 history/tool 配额只写报告不生效；
  - F3 多线程落盘 database is locked；
  - F7 ref 里的 ../../ 可写出 artifact 目录；
  - F12 _safe_path 字符串前缀判断放过 /tmp/work-evil。
"""
from __future__ import annotations

import contextlib
import shutil
import tempfile
import threading
import unittest
from pathlib import Path

from ch.budget import TokenBudget, build_groups
from ch.checkpoint import CheckpointStore
from ch.compress import ToolOutputCompressor
from ch.memory import MemoryManager
from ch.tokens import estimate, estimate_messages
from ch.tools import _safe_path, default_registry
from ch.types import Message, ToolCall


@contextlib.contextmanager
def tmpdir():
    d = Path(tempfile.mkdtemp())
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


class TestBudgetQuota(unittest.TestCase):
    def test_cjk_memory_respects_token_budget(self):
        """1600 个汉字塞进 total=800 的预算，结果必须真的 <= 800 token。"""
        b = TokenBudget(total=800)
        out, rep = b.assemble(
            [Message(role="system", content="system")],
            "汉" * 1600,
            [Message(role="user", content="你好")],
        )
        self.assertTrue(rep.memory_truncated)
        self.assertLessEqual(rep.tokens_after, 800)
        self.assertLessEqual(estimate_messages(out), 800)

    def test_tool_quota_actually_trims(self):
        """工具区配额必须真正限制 tool 输出，而不是只写进报告。"""
        b = TokenBudget(total=600, tool_share=0.45, history_share=0.35, memory_share=0.20)
        call = ToolCall(id="c1", name="grep", arguments={})
        history = [
            Message(role="assistant", content="", tool_calls=[call]),
            Message(role="tool", content="汉" * 3000, tool_call_id="c1", name="grep"),
        ]
        out, rep = b.assemble([Message(role="system", content="sys")], "", history)
        tool_tokens = estimate_messages([m for m in out if m.role == "tool"])
        self.assertGreater(rep.tool_quota, 0)
        self.assertLessEqual(tool_tokens, rep.tool_quota + 8)
        self.assertTrue(rep.hard_truncated)

    def test_groups_keep_tool_pairing(self):
        call = ToolCall(id="c1", name="grep", arguments={})
        gs = build_groups(
            [
                Message(role="assistant", content="", tool_calls=[call]),
                Message(role="tool", content="r", tool_call_id="c1"),
            ]
        )
        self.assertEqual(len(gs), 1)
        self.assertEqual(len(gs[0]), 2)


class TestCompressBudget(unittest.TestCase):
    def test_cjk_and_ascii_fit_budget(self):
        with tmpdir() as d:
            c = ToolOutputCompressor(d)
            for text in ("汉" * 5000, "a" * 5000, "汉ab" * 800):
                for budget in (40, 100, 300):
                    r = c.compress("c1", "grep", text, max_tokens=budget)
                    self.assertLessEqual(r.final_tokens, budget, f"{len(text)} chars @ {budget}")

    def test_ref_cannot_escape_artifact_dir(self):
        with tmpdir() as d:
            c = ToolOutputCompressor(d)
            r = c.compress("../../evil", "grep", "x" * 5000, max_tokens=10)
            self.assertNotIn("/", r.ref)
            self.assertNotIn("\\", r.ref)
            files = {p.name for p in (Path(d) / "tool_outputs").iterdir()}
            self.assertIn(f"{r.ref}.txt", files)
            self.assertFalse((Path(d).parent / "evil.txt").exists())
            self.assertIsNone(c.read_ref("../../evil"))
            self.assertIsNone(c.read_ref("../tool_outputs/grep"))
            self.assertIsNotNone(c.read_ref(r.ref))


class TestSandbox(unittest.TestCase):
    def test_sibling_prefix_is_rejected(self):
        with tmpdir() as d:
            root = d / "work"
            (d / "work-evil").mkdir()
            root.mkdir()
            with self.assertRaises(PermissionError):
                _safe_path(root, str(d / "work-evil" / "x"))
            self.assertEqual(_safe_path(root, "a/b.txt"), (root / "a/b.txt").resolve())

    def test_default_registry_has_no_shell(self):
        reg = default_registry(root=".")
        self.assertNotIn("shell", reg.names())
        self.assertIn("shell", default_registry(root=".", include_shell=True).names())


class TestCheckpointConcurrency(unittest.TestCase):
    def test_parallel_writes_do_not_lock(self):
        with tmpdir() as d:
            cp = CheckpointStore(d / "cp.sqlite3")
            errors: list[str] = []

            def worker(i: int) -> None:
                try:
                    for step in range(15):
                        cp.save(f"run{i}", step, "task", [], [])
                except Exception as e:  # noqa: BLE001 - 测试里要把异常收集回来断言
                    errors.append(repr(e))

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(errors, [])
            self.assertEqual(len(cp.list_runs(100)), 8)
            cp.close()
            cp.close()  # 幂等
            self.assertTrue(cp.closed)


class TestPackageSurface(unittest.TestCase):
    def test_top_level_imports(self):
        """`from ch import Agent` 必须可用，且版本号与 pyproject 保持一致。"""
        import re

        import ch

        self.assertTrue(callable(ch.Agent))
        self.assertIn("Agent", ch.__all__)
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        text = pyproject.read_text(encoding="utf-8")
        self.assertRegex(text, rf'version = "{re.escape(ch.__version__)}"')


class TestMemoryTruncation(unittest.TestCase):
    def test_recall_block_respects_budget(self):
        with tmpdir() as d:
            mem = MemoryManager(d / "m.sqlite3")
            for i in range(8):
                mem.remember_fact(f"键{i}", "值" * 300)
            block = mem.recall_block("键", budget_tokens=100)
            self.assertLessEqual(estimate(block), 100)
            self.assertIn("memory truncated", block)
            mem.close()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
