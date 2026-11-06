"""离线演示：不需要任何 API key，直接 `python examples/run_mock.py`。

演示四件事：
  1. 工具循环 + 每步 trace（token / 成本 / 延迟）
  2. 预算裁剪：上下文塞不下时砍了什么、砍了几组
  3. 三层记忆：写入 -> 去重 reinforce -> 冲突 supersede -> 召回
  4. 工具输出压缩：200KB 输出压成 1KB，全文归档可回查
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ch.agent import Agent  # noqa: E402
from ch.budget import TokenBudget  # noqa: E402
from ch.compress import ToolOutputCompressor  # noqa: E402
from ch.config import Config  # noqa: E402
from ch.llm import Router, ScriptedLLM  # noqa: E402
from ch.memory import MemoryManager  # noqa: E402
from ch.tools import default_registry  # noqa: E402
from ch.trace import Tracer  # noqa: E402
from ch.types import Message, ToolCall  # noqa: E402


def demo_loop() -> None:
    print("=" * 70)
    print("1) 工具循环 + trace（ScriptedLLM 模拟模型决策）")
    print("=" * 70)
    cfg = Config(context_budget=8000, db_path=str(ROOT / ".artifacts/demo.sqlite3"),
                 artifact_dir=str(ROOT / ".artifacts"))
    cfg.ensure_dirs()
    script = ScriptedLLM(
        [
            {"tool": "calculator", "args": {"expr": "(128 + 64) * 3"}},
            {"tool": "calculator", "args": {"expr": "576 / 40"}},
            {"text": "总工作量 576，摊到 40 步每步 14.4。"},
        ]
    )
    agent = Agent(
        router=Router(cfg, main=script, utility=script),
        tools=default_registry(root=str(ROOT), whitelist={"calculator", "read_file", "list_dir"}),
        cfg=cfg,
        tracer=Tracer(),
    )
    res = agent.run("算 (128+64)*3，再除以 40，说明每一步。")
    print("答案：", res.answer)
    print("停机原因：", res.stop_reason, "| 步数：", res.steps)
    print(agent.tracer.table())
    print()


def demo_budget() -> None:
    print("=" * 70)
    print("2) 预算裁剪：超出 8000 token 时先砍谁")
    print("=" * 70)
    budget = TokenBudget(total=3000)
    pinned = [
        Message(role="system", content="你是 agent。" * 20, pinned=True),
        Message(role="user", content="分析这个项目的上下文管理策略。", pinned=True),
    ]
    history: list[Message] = []
    for i in range(12):
        call = ToolCall(id=f"c{i}", name="read_file", arguments={"path": f"file{i}.py"})
        history.append(Message(role="assistant", content="", tool_calls=[call]))
        history.append(Message(role="tool", content="代码内容 " * 400, tool_call_id=f"c{i}", name="read_file"))
    memory_block = "【已知事实】\n- token 预算: 8000 token\n" * 10

    out, rep = budget.assemble(pinned, memory_block, history)
    print("裁剪前 token：", rep.tokens_before)
    print("裁剪后 token：", rep.tokens_after, f"/ 预算 {rep.budget}")
    print("丢弃组数：", rep.dropped_groups)
    print("记忆被截断：", rep.memory_truncated, "| 工具输出硬截断：", rep.hard_truncated)
    print("保留消息角色序列：", [m.role for m in out])
    print()


def demo_memory() -> None:
    print("=" * 70)
    print("3) 三层记忆：去重 / 冲突更新 / 时间衰减 / 召回")
    print("=" * 70)
    # 每次从空库开始，否则第二次跑会看到 superseded 而不是第一次的 inserted
    db = ROOT / ".artifacts/demo-memory.sqlite3"
    db.parent.mkdir(parents=True, exist_ok=True)
    if db.exists():
        db.unlink()
    mem = MemoryManager(str(db), context_budget=3000)
    print("写入：", mem.remember_fact("默认上下文预算", "8000 token", source="user"))
    print("重复写入同值：", mem.remember_fact("默认上下文预算", "8000 token", source="agent"))
    print("写入冲突值：", mem.remember_fact("默认上下文预算", "2500 token", source="user"))
    mem.record_episode("用户希望优化 token 成本，给出了 34% 的下降数据。")
    mem.record_episode("设计要点：MCP 与 Skill 的区别。")
    print("\n召回（query=上下文预算 多少钱）：")
    print(mem.recall_block("上下文预算", budget_tokens=400))
    print("\n统计：", mem.stats())
    print()


def demo_compress() -> None:
    print("=" * 70)
    print("4) 工具输出压缩：超大输出 -> 头尾保留 + 归档可回查")
    print("=" * 70)
    comp = ToolOutputCompressor(str(ROOT / ".artifacts"))
    big = "\n".join(f"line {i}: " + "x" * 80 for i in range(2000))
    res = comp.compress("call_9", "read_file", big, max_tokens=400)
    print("原始字符数：", len(big), "| 压缩后字符数：", len(res.text))
    print("省略字符数：", res.omitted_chars, "| 归档 ref：", res.ref)
    back = comp.read_ref(res.ref)
    print("归档可完整回查：", back is not None and len(back) == len(big))
    print("压缩后片段开头：", res.text[:80].replace("\n", " | "))
    print()


if __name__ == "__main__":
    demo_loop()
    demo_budget()
    demo_memory()
    demo_compress()
