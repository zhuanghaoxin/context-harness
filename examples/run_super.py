"""Super-agent v2 离线对照实验：单 loop 串行  vs  Filter+Router+WorkerPool 双层。

    python examples/run_super.py

用 DemoLLM 替代真实模型（离线可跑、结果可复现），但**延迟是真睡眠**，所以
makespan / P95 的对比是有意义的：它测的是"机制让多少活可以同时进行"，
不是"模型快了多少"。真实模型的判断质量必须 `--online` 才有结论。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ch.agent import Agent  # noqa: E402
from ch.config import Config  # noqa: E402
from ch.llm import Router, ScriptedLLM  # noqa: E402
from ch.supervisor import MODE_SINGLE, MODE_SUPER, Supervisor, compare  # noqa: E402
from ch.tools import default_registry  # noqa: E402
from ch.types import LLMResponse, Message, ToolCall, Usage, new_id  # noqa: E402

# ------------------------------------------------------------------ 测试输入
# 6 条真实任务 + 4 条噪音。噪音在 single 模式下会被完整跑一遍（纯浪费）。
ITEMS: list[tuple[str, str | None, str]] = [
    ("t1", "login.py", "帮我修复 login.py 里登录失败的报错"),
    ("t2", "sales.csv", "分析 sales.csv 里订单金额的异常值"),
    ("t3", "api_client.py", "优化 api_client.py 的重试逻辑，超时太多"),
    ("t4", "docs/architecture.md", "整理 docs/architecture.md 的上下文管理章节"),
    ("t5", "mem0", "对比 mem0 和 zep 的记忆方案差异"),
    ("t6", "scheduler.py", "排查 scheduler.py 任务堆积的根因"),
    ("n1", None, "你好"),
    ("n2", None, "嗯"),
    ("n3", None, "GET /api/health 200 3ms"),
    ("n4", None, "2026-11-04 10:23:11 INFO worker heartbeat ok"),
]


class DemoLLM(ScriptedLLM):
    """固定两步行为：先算一次数（走工具），再回一句包含任务前缀的结论。

    故意不用 ScriptedLLM 的脚本列表：两种模式下 Agent 被调用的次序不同，
    按"最后一条 user 消息"应答才能让两个实验臂拿到同样的任务。
    """

    def __init__(self, latency_s: float = 0.0, price: tuple[float, float] = (0.15, 0.60)):
        super().__init__([{"text": "noop"}], model_name="demo-llm", latency_s=latency_s)
        self.price = price  # USD / 1M tokens，与 cfg.price_per_mtok["demo-llm"] 对齐

    def _resp(self, msg: Message, p: int, c: int) -> LLMResponse:
        cost = (p * self.price[0] + c * self.price[1]) / 1_000_000
        return LLMResponse(message=msg, usage=Usage(p, c), model=self.model_name, cost_usd=cost)

    def complete(self, messages, tools=None):
        self.calls += 1
        self.seen.append(list(messages))
        if self.latency_s > 0:
            time.sleep(self.latency_s)

        last_user = ""
        for m in reversed(messages):
            if m.role == "user":
                last_user = (m.content or "").strip()
                break

        if not any(m.role == "tool" for m in messages):
            call = ToolCall(id=new_id("call"), name="calculator", arguments={"expr": "(1200+800)*1.05"})
            return self._resp(Message(role="assistant", content="", tool_calls=[call]), 180, 30)

        return self._resp(Message(role="assistant", content=f"已完成：{last_user[:24]}"), 220, 40)


def make_cfg(latency_s: float, max_workers: int = 4) -> Config:
    cfg = Config.from_env()
    cfg.context_budget = 3000
    cfg.max_steps = 6
    cfg.max_workers = max_workers
    cfg.sim_latency_s = latency_s
    cfg.db_path = ".artifacts/super.db"
    cfg.artifact_dir = ".artifacts/super"
    cfg.price_per_mtok["demo-llm"] = (0.15, 0.60)  # 让离线实验也能报出成本
    cfg.ensure_dirs()
    return cfg


def make_agent(cfg: Config) -> Agent:
    llm = DemoLLM(latency_s=cfg.sim_latency_s, price=cfg.price_per_mtok.get("demo-llm", (0.15, 0.60)))
    return Agent(
        router=Router(cfg, main=llm, utility=llm),
        tools=default_registry(root=".", approval=lambda n, a: False, whitelist={"calculator"}),
        cfg=cfg,
        memory=None,
        tracer=None,
    )


def make_online_agent(cfg: Config) -> Agent:
    """真实模型：Router 自己按 role 建 OpenAICompatLLM（main / utility 两个模型）。"""
    return Agent(
        router=Router(cfg),
        tools=default_registry(root=".", approval=lambda n, a: False, whitelist={"calculator"}),
        cfg=cfg,
        memory=None,
        tracer=None,
    )


def build_items() -> tuple[list, dict]:
    from ch.supervisor import WorkItem

    items, checks = [], {}
    for iid, token, text in ITEMS:
        items.append(WorkItem(id=iid, text=text))
        if token:
            checks[iid] = {"type": "contains", "value": token}
    return items, checks


def main() -> int:
    import sys

    online = "--online" in sys.argv
    latency = 0.0 if online else 0.12  # 离线时每次模型调用睡 120ms，让并行收益可观测
    cfg = make_cfg(latency)
    if online:
        cfg = Config.from_env()
        cfg.context_budget = 3000
        cfg.max_steps = 6
        cfg.max_workers = 4
        cfg.db_path = ".artifacts/super-online.db"
        cfg.artifact_dir = ".artifacts/super-online"
        cfg.ensure_dirs()
        if cfg.offline:
            print("未配置真实模型：请在 .env 里填 CH_BASE_URL / CH_API_KEY / CH_MODEL 后重试。")
            print("离线演示请去掉 --online。")
            return 2

    factory = (lambda: make_online_agent(cfg)) if online else (lambda: make_agent(cfg))
    checks_all = {} if online else build_items()[1]
    if online:
        # 真实模型给自由文本，只能判"有没有给出实质回答"
        checks_all = {iid: {"type": "min_len", "value": 30} for iid, _, _ in ITEMS if iid.startswith("t")}
    sup = Supervisor(cfg, agent_factory=factory, checks=checks_all)

    print("=" * 78)
    mode = "在线，真实模型" if online else "离线，模型调用模拟 %dms" % int(latency * 1000)
    print("Super-agent v2 对照实验（%s）" % mode)
    print("输入：6 条真实任务 + 4 条噪音消息")
    print("=" * 78)

    # 两个实验臂各自从干净的 item 状态开始
    items_single, checks = build_items()
    sup.checks = checks_all
    r_single = sup.run_single(items_single)
    print("\n" + r_single.table())

    items_super, checks = build_items()
    sup.checks = checks_all
    r_super = sup.run_super(items_super)
    print("\n" + r_super.table())

    print("\n" + "=" * 78)
    print("对照（以 single 为基准）")
    print("=" * 78)
    rows = compare([r_single, r_super])
    keys = [
        ("submitted", "提交消息"),
        ("processed", "实跑 loop"),
        ("completed", "完成"),
        ("rejected", "过滤拒收"),
        ("pass_rate", "通过率"),
        ("makespan_ms", "makespan(ms)"),
        ("p95_ms", "P95 延迟(ms)"),
        ("peak_concurrency", "峰值并发"),
        ("proactive", "主动委派"),
        ("tokens", "总 token"),
        ("cost_usd", "总成本($)"),
    ]
    for k, label in keys:
        a, b = rows[0][k], rows[1][k]
        delta = rows[1].get(f"d_{k}", "")
        print(f"  {label:<14} {str(a):>10} -> {str(b):>10}   {delta}")

    print("\n--- 过滤层（规则 vs 模型）---")
    print(" ", r_super.filter_stats or {"note": "single 模式没有过滤器"})

    print("\n--- 每条消息去向 ---")
    print(f"  {'id':<4} {'single':<10} {'super':<10} {'super 备注'}")
    for a, b in zip(r_single.items, r_super.items):
        print(f"  {a.id:<4} {a.status:<10} {b.status:<10} {b.note or b.stop_reason}")

    print("\n结论要点：")
    print("  1) 4 条噪音在 single 模式里被完整跑了 %d 个 loop，而 super 模式 0 成本丢弃"
          % sum(1 for i in r_single.items if i.id.startswith("n")))
    print("  2) makespan 的差距只来自并行度，不来自模型速度")
    print("  3) 真实数字要 --online 换成 OpenAICompatLLM 重跑，这里测的是机制不是模型")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
