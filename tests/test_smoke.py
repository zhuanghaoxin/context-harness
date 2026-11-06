"""单元测试：覆盖每个自研模块的核心行为。

跑法（项目根目录，不需要设置 PYTHONPATH）：
    python -m unittest discover -s tests -t . -v
或：
    python -m pytest -q
"""
from __future__ import annotations

import contextlib
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from ch.agent import Agent  # noqa: E402
from ch.budget import TokenBudget, build_groups  # noqa: E402
from ch.compress import ToolOutputCompressor  # noqa: E402
from ch.config import Config  # noqa: E402
from ch.dag import DAG, Node  # noqa: E402
from ch.evaluate import check  # noqa: E402
from ch.llm import Router, ScriptedLLM  # noqa: E402
from ch.memory import HashingEmbedder, MemoryManager, SemanticMemory, cosine  # noqa: E402
from ch.skill import Skill, SkillRegistry, default_skills  # noqa: E402
from ch.tokens import estimate, estimate_messages  # noqa: E402
from ch.tools import default_registry  # noqa: E402
from ch.trace import Tracer  # noqa: E402
from ch.types import Message, ToolCall  # noqa: E402


@contextlib.contextmanager
def tmpdir():
    """Windows 上 sqlite 句柄会让 TemporaryDirectory 清理失败，这里忽略清理错误。"""
    d = Path(tempfile.mkdtemp())
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def make_cfg(tmp: Path, budget: int = 4000) -> Config:
    cfg = Config(context_budget=budget, db_path=str(tmp / "h.sqlite3"), artifact_dir=str(tmp / "art"))
    cfg.ensure_dirs()
    return cfg


class TestTokens(unittest.TestCase):
    def test_cjk_heavier_than_ascii(self):
        self.assertGreater(estimate("你好世界"), estimate("abcd"))
        self.assertGreater(estimate("你好世界"), 0)
        self.assertEqual(estimate(""), 0)

    def test_messages_include_overhead(self):
        msgs = [Message(role="user", content="hi")]
        self.assertGreater(estimate_messages(msgs), estimate("hi"))


class TestBudget(unittest.TestCase):
    def test_groups_keep_tool_pairing(self):
        history = [
            Message(role="assistant", content="", tool_calls=[ToolCall("c1", "t", {})]),
            Message(role="tool", content="r", tool_call_id="c1"),
            Message(role="assistant", content="done"),
        ]
        groups = build_groups(history)
        self.assertEqual(len(groups), 2)
        self.assertEqual([m.role for m in groups[0]], ["assistant", "tool"])

    def test_trim_drops_oldest_and_keeps_last(self):
        budget = TokenBudget(total=400)
        pinned = [Message(role="user", content="目标" * 5, pinned=True)]
        history = []
        for i in range(10):
            history.append(Message(role="assistant", content=f"step {i} " + "内" * 200))
        out, rep = budget.assemble(pinned, "", history)
        self.assertGreater(rep.dropped_groups, 0)
        self.assertLessEqual(rep.tokens_after, 450)
        self.assertIn("step 9", out[-1].content)

    def test_no_orphan_tool_messages(self):
        budget = TokenBudget(total=600)
        pinned = [Message(role="user", content="t", pinned=True)]
        history = [
            Message(role="assistant", content="", tool_calls=[ToolCall("c1", "x", {})]),
            Message(role="tool", content="y" * 300, tool_call_id="c1"),
            Message(role="assistant", content="", tool_calls=[ToolCall("c2", "x", {})]),
            Message(role="tool", content="z" * 300, tool_call_id="c2"),
        ]
        out, _ = budget.assemble(pinned, "", history)
        declared = {c.id for m in out for c in m.tool_calls}
        for m in out:
            if m.role == "tool":
                self.assertIn(m.tool_call_id, declared)

    def test_memory_truncated_when_over_quota(self):
        budget = TokenBudget(total=300)
        pinned = [Message(role="user", content="t", pinned=True)]
        out, rep = budget.assemble(pinned, "事实" * 2000, [])
        self.assertTrue(rep.memory_truncated)
        self.assertTrue(any(m.name == "recalled_memory" for m in out))


class TestCompress(unittest.TestCase):
    def test_small_output_untouched(self):
        with tmpdir() as d:
            c = ToolOutputCompressor(d)
            r = c.compress("c", "t", "short", 100)
            self.assertEqual(r.text, "short")
            self.assertIsNone(r.ref)

    def test_big_output_archived_and_retrievable(self):
        with tmpdir() as d:
            c = ToolOutputCompressor(d)
            big = "行内容" * 5000
            r = c.compress("c1", "grep", big, 100)
            self.assertGreater(r.omitted_chars, 0)
            self.assertIsNotNone(r.ref)
            self.assertEqual(c.read_ref(r.ref), big)
            self.assertLess(len(r.text), len(big))


class TestTools(unittest.TestCase):
    def test_missing_required_arg_returns_structured_error(self):
        reg = default_registry(root=".", whitelist={"calculator"})
        res = reg.call("calculator", {})
        self.assertFalse(res.ok)
        self.assertIn("参数校验失败", res.error)

    def test_unknown_tool(self):
        reg = default_registry(root=".", whitelist={"calculator"})
        res = reg.call("rm_rf", {})
        self.assertFalse(res.ok)
        self.assertIn("不在可用列表", res.error)

    def test_whitelist_filtering(self):
        reg = default_registry(root=".", whitelist={"calculator"})
        self.assertEqual(reg.names(), ["calculator"])
        self.assertEqual(len(reg.schemas()), 1)

    def test_dangerous_tool_denied_by_default(self):
        reg = default_registry(root=".", whitelist={"write_file"})
        res = reg.call("write_file", {"path": "x.txt", "content": "hi"})
        self.assertFalse(res.ok)
        self.assertIn("未获审批", res.error)

    def test_path_escape_blocked(self):
        reg = default_registry(root=".", whitelist={"read_file"})
        res = reg.call("read_file", {"path": "../../etc/passwd"})
        self.assertFalse(res.ok)

    def test_calculator_rejects_code(self):
        reg = default_registry(root=".", whitelist={"calculator"})
        self.assertTrue(reg.call("calculator", {"expr": "2*3"}).ok)
        self.assertFalse(reg.call("calculator", {"expr": "__import__('os')"}).ok)

    def test_type_validation(self):
        reg = default_registry(root=".", whitelist={"calculator"})
        self.assertFalse(reg.call("calculator", {"expr": 123}).ok)


class TestMemory(unittest.TestCase):
    def test_embedder_self_similarity(self):
        e = HashingEmbedder()
        self.assertAlmostEqual(cosine(e.embed("上下文预算"), e.embed("上下文预算")), 1.0, places=5)
        self.assertLess(cosine(e.embed("上下文预算"), e.embed("完全无关的内容")), 0.9)

    def test_dedupe_reinforces(self):
        with tmpdir() as d:
            m = SemanticMemory(d / "m.sqlite3")
            self.assertEqual(m.remember("k", "v1"), "inserted")
            self.assertEqual(m.remember("K ", " v1"), "reinforced")
            facts = m.all_valid()
            self.assertEqual(len(facts), 1)
            self.assertGreater(facts[0].confidence, 0.7)
            m.close()

    def test_conflict_supersedes(self):
        with tmpdir() as d:
            m = SemanticMemory(d / "m.sqlite3")
            m.remember("预算", "8000")
            self.assertEqual(m.remember("预算", "2500"), "superseded")
            self.assertEqual([f.value for f in m.all_valid()], ["2500"])
            m.close()

    def test_decay_lowers_score(self):
        import time as _t

        with tmpdir() as d:
            m = SemanticMemory(d / "m.sqlite3", half_life_days=1)
            m.remember("预算", "8000")
            now = _t.time()
            fresh = m.recall("预算", now=now)[0].score
            old = m.recall("预算", now=now + 86400 * 7)[0].score
            self.assertLess(old, fresh)
            m.close()

    def test_manager_recall_block(self):
        with tmpdir() as d:
            mem = MemoryManager(d / "m.sqlite3")
            mem.remember_fact("项目预算", "8000 token")
            mem.record_episode("用户强调过成本问题")
            block = mem.recall_block("预算", budget_tokens=300)
            self.assertIn("8000 token", block)
            self.assertIn("已知事实", block)
            mem.close()

    def test_episodic_search(self):
        with tmpdir() as d:
            mem = MemoryManager(d / "m.sqlite3")
            mem.record_episode("今天研究了 MCP 协议的握手流程")
            mem.record_episode("周末去爬山")
            hits = mem.episodic.search("MCP 握手", k=1)
            self.assertEqual(len(hits), 1)
            self.assertIn("MCP", hits[0].text)
            mem.close()


class TestAgentLoop(unittest.TestCase):
    def test_final_answer_stops(self):
        with tmpdir() as d:
            cfg = make_cfg(d)
            llm = ScriptedLLM([{"text": "done"}])
            a = Agent(Router(cfg, main=llm, utility=llm), default_registry(root=".", whitelist={"calculator"}), cfg)
            res = a.run("hi")
            self.assertEqual(res.stop_reason, "final")
            self.assertEqual(res.steps, 1)
            self.assertEqual(res.answer, "done")

    def test_tool_then_final(self):
        with tmpdir() as d:
            cfg = make_cfg(d)
            llm = ScriptedLLM([{"tool": "calculator", "args": {"expr": "2+2"}}, {"text": "4"}])
            a = Agent(Router(cfg, main=llm, utility=llm), default_registry(root=".", whitelist={"calculator"}), cfg)
            res = a.run("2+2")
            self.assertEqual(res.stop_reason, "final")
            self.assertEqual(res.steps, 2)
            self.assertEqual(len(res.trace), 2)
            self.assertIn("4", res.answer)

    def test_repeated_tool_guard(self):
        with tmpdir() as d:
            cfg = make_cfg(d)
            cfg.repeat_limit = 2
            llm = ScriptedLLM([{"tool": "calculator", "args": {"expr": "1+1"}}])
            a = Agent(Router(cfg, main=llm, utility=llm), default_registry(root=".", whitelist={"calculator"}), cfg)
            res = a.run("loop")
            self.assertEqual(res.stop_reason, "repeated_tool")

    def test_max_steps_guard(self):
        with tmpdir() as d:
            cfg = make_cfg(d)
            cfg.max_steps = 3
            scripts = [{"tool": "calculator", "args": {"expr": str(i)}} for i in range(5)]
            llm = ScriptedLLM(scripts)
            a = Agent(Router(cfg, main=llm, utility=llm), default_registry(root=".", whitelist={"calculator"}), cfg)
            res = a.run("never finish")
            # 现在多了一个更早的停机路径：Watchdog 发现连续同样的工具调用就中断
            self.assertIn(res.stop_reason, {"max_steps", "repeated_tool", "watchdog:loop"})
            self.assertLessEqual(res.steps, 3)

    def test_memory_tool_installed_and_writes(self):
        with tmpdir() as d:
            cfg = make_cfg(d)
            mem = MemoryManager(cfg.db_path)
            llm = ScriptedLLM([{"tool": "remember_fact", "args": {"key": "k", "value": "v"}}, {"text": "ok"}])
            a = Agent(Router(cfg, main=llm, utility=llm), default_registry(root=".", whitelist={"calculator"}), cfg, memory=mem)
            a.run("记住")
            self.assertEqual(len(mem.semantic.all_valid()), 1)
            mem.close()

    def test_checkpoint_save_and_resume(self):
        from ch.checkpoint import CheckpointStore

        with tmpdir() as d:
            cfg = make_cfg(d)
            cp = CheckpointStore(cfg.db_path)
            llm = ScriptedLLM([{"tool": "calculator", "args": {"expr": "1+1"}}, {"text": "2"}])
            a = Agent(
                Router(cfg, main=llm, utility=llm),
                default_registry(root=".", whitelist={"calculator"}),
                cfg,
                checkpoint=cp,
            )
            res = a.run("x", run_id="r1")
            snap = cp.load("r1")
            self.assertIsNotNone(snap)
            self.assertGreaterEqual(snap.step, 1)
            self.assertTrue(any(m.role == "tool" for m in snap.history))
            self.assertEqual(len(cp.list_runs()), 1)
            self.assertEqual(res.run_id, "r1")
            cp.close()

    def test_trace_summary(self):
        with tmpdir() as d:
            cfg = make_cfg(d)
            llm = ScriptedLLM([{"text": "ok"}])
            t = Tracer()
            a = Agent(Router(cfg, main=llm, utility=llm), default_registry(root=".", whitelist={"calculator"}), cfg, tracer=t)
            a.run("x")
            s = t.summary()
            self.assertEqual(s["steps"], 1)
            self.assertIn("tokens_per_step", s)


class TestDAG(unittest.TestCase):
    def test_topological_layers(self):
        dag = (
            DAG("t")
            .add(Node("a", "A"))
            .add(Node("b", "B", deps=("a",)))
            .add(Node("c", "C", deps=("a",)))
            .add(Node("d", "D", deps=("b", "c")))
        )
        self.assertEqual(dag.layers(), [["a"], ["b", "c"], ["d"]])

    def test_cycle_detected(self):
        dag = DAG("t").add(Node("a", "A", deps=("b",))).add(Node("b", "B", deps=("a",)))
        with self.assertRaises(ValueError):
            dag.validate()

    def test_missing_dep_detected(self):
        dag = DAG("t").add(Node("a", "A", deps=("zzz",)))
        with self.assertRaises(ValueError):
            dag.validate()

    def test_run_with_scripted_nodes(self):
        from ch.checkpoint import CheckpointStore

        with tmpdir() as d:
            cfg = make_cfg(d)
            cp = CheckpointStore(cfg.db_path)
            dag = DAG("demo").add(Node("a", "第一步")).add(Node("b", "第二步", deps=("a",)))

            def factory(node):
                llm = ScriptedLLM([{"text": f"{node.name}-ok"}])
                return Agent(
                    Router(cfg, main=llm, utility=llm),
                    default_registry(root=".", whitelist={"calculator"}),
                    cfg,
                )

            out = dag.run(factory, checkpoint=cp)
            self.assertEqual(out["outputs"]["a"], "a-ok")
            self.assertEqual(out["outputs"]["b"], "b-ok")
            self.assertEqual(cp.node_status("demo")["b"][0], "completed")
            # 二次运行：completed 节点应被跳过（不产生新 run）
            out2 = dag.run(factory, checkpoint=cp, resume=True)
            self.assertEqual(out2["results"], {})
            cp.close()


class TestSkill(unittest.TestCase):
    def test_dependency_gap_reported(self):
        reg = default_skills()
        self.assertEqual(sorted(reg.validate({"read_file", "list_dir", "calculator"})), [])
        self.assertIn("data_check", reg.validate({"read_file"}))

    def test_prompt_block_excludes_broken_skill(self):
        reg = default_skills()
        block = reg.prompt_block({"read_file", "list_dir"})
        self.assertIn("code_review", block)
        self.assertNotIn("data_check", block)

    def test_from_mcp_tool(self):
        reg = SkillRegistry()
        s = reg.from_mcp_tool(
            {"name": "search", "description": "d", "inputSchema": {"properties": {"q": {"type": "string"}}}}
        )
        self.assertEqual(s.name, "search")
        self.assertIn("q", s.inputs)
        self.assertEqual(s.tools, ("search",))

    def test_four_elements_present(self):
        s = Skill(name="x", description="d", inputs={"a": "string"}, outputs={"o": "string"}, tools=("t",))
        block = s.to_prompt_block()
        for key in ("输入", "输出", "依赖工具"):
            self.assertIn(key, block)


class TestEvaluate(unittest.TestCase):
    def test_checkers(self):
        self.assertTrue(check("结果 576", {"type": "numeric", "value": [576]}))
        self.assertFalse(check("结果 577", {"type": "numeric", "value": [576]}))
        self.assertTrue(check("降幅 35.4%", {"type": "contains", "value": "35.4"}))
        self.assertTrue(check("abc", {"type": "regex", "value": "^a.c$"}))
        self.assertTrue(
            check(
                "62.5 和 77.5",
                {
                    "type": "all",
                    "value": [
                        {"type": "contains", "value": "62.5"},
                        {"type": "contains", "value": "77.5"},
                    ],
                },
            )
        )

    def test_unknown_checker_raises(self):
        with self.assertRaises(ValueError):
            check("x", {"type": "nope"})

    def test_task_file_is_valid(self):
        from ch.evaluate import load_tasks

        tasks = load_tasks(ROOT / "tasks" / "tasks.json")
        self.assertGreaterEqual(len(tasks), 8)
        for t in tasks:
            self.assertTrue(t.script, f"{t.id} 缺少离线 script")

    def test_variant_configs_differ(self):
        from ch.evaluate import make_config

        b = make_config("baseline")
        f = make_config("full")
        self.assertGreater(b.context_budget, f.context_budget)
        self.assertNotEqual(b.db_path, f.db_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
