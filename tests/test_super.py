"""Super-agent v2 的测试：过滤 / Watchdog / Router-Worker / 两模式对照。

跑法（项目根目录，不需要设置 PYTHONPATH）：
    python -m unittest discover -s tests -t .
"""
from __future__ import annotations

import contextlib
import shutil
import tempfile
import time
import unittest

from ch.config import Config
from ch.filter import ACT_NOW, DEFER, IGNORE, MessageFilter, rule_score
from ch.llm import Router, ScriptedLLM
from ch.router_worker import (
    NEW_WORKER,
    REJECT,
    REUSE_WORKER,
    TaskRouter,
    WorkItem,
    WorkerPool,
)
from ch.supervisor import MODE_SINGLE, MODE_SUPER, Supervisor
from ch.tools import default_registry
from ch.agent import Agent
from ch.watchdog import ABORT, OK, REMIND_GOAL, Watchdog, anchors, overlap


@contextlib.contextmanager
def tmpdir():
    d = tempfile.mkdtemp(prefix="ch-super-")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def make_cfg(**kw) -> Config:
    cfg = Config.from_env()
    # 单测必须与本地 .env 无关：否则填了真实 key/embedding 后，
    # 这里的 Config 会去真调远端（实测把首个 submit 拖慢 774ms，把并行优势测没了）。
    cfg.base_url = "http://stub"
    cfg.api_key = ""
    cfg.embedding_model = ""
    cfg.context_budget = 3000
    cfg.max_steps = 4
    for k, v in kw.items():
        setattr(cfg, k, v)
    cfg.price_per_mtok["echo-llm"] = (0.15, 0.60)
    return cfg


def echo_agent(cfg: Config, latency_s: float = 0.0) -> Agent:
    """一步就收尾的 Agent：用于测机制，不测推理。"""
    llm = ScriptedLLM([{"text": "done"}], model_name="echo-llm", latency_s=latency_s)
    return Agent(
        router=Router(cfg, main=llm, utility=llm),
        tools=default_registry(root=".", approval=lambda n, a: False, whitelist={"calculator"}),
        cfg=cfg,
        memory=None,
    )


class TestRuleFilter(unittest.TestCase):
    def test_noise_is_flagged(self):
        for text in ["你好", "嗯", "收到!", "", "GET /api/health 200", "2026-11-04 10:23:11 INFO x"]:
            r = rule_score(text)
            self.assertTrue(r.noise, f"应当判为噪音：{text!r}")

    def test_real_task_scores_high(self):
        r = rule_score("帮我修复 login.py 里登录失败的报错")
        self.assertFalse(r.noise)
        self.assertGreaterEqual(r.score, 0.70)

    def test_noise_dropped_without_model_call(self):
        f = MessageFilter(make_cfg(), router=None)
        d = f.decide("你好")
        self.assertEqual(d.action, IGNORE)
        self.assertEqual(d.stage, "rule")
        self.assertEqual(f.stats.model_called, 0)  # 关键：噪音不花钱

    def test_clear_task_also_skips_model(self):
        f = MessageFilter(make_cfg(), router=None)
        d = f.decide("帮我修复 login.py 里登录失败的报错")
        self.assertEqual(d.action, ACT_NOW)
        self.assertEqual(d.stage, "rule")
        self.assertEqual(f.stats.model_called, 0)

    def test_gray_zone_fails_open(self):
        # 灰区拿不到模型判断时应当放行：漏任务的代价 >> 多处理一条噪音
        f = MessageFilter(make_cfg(), router=None)
        d = f.decide("对比 mem0 和 zep 的记忆方案差异")  # 落在 (low, high) 灰区
        self.assertEqual(d.stage, "model")
        self.assertEqual(d.action, ACT_NOW)
        self.assertEqual(f.stats.by_action.get(ACT_NOW), 1)

    def test_filter_can_be_disabled(self):
        f = MessageFilter(make_cfg(filter_enabled=False), router=None)
        self.assertEqual(f.decide("你好").action, ACT_NOW)


class TestWatchdog(unittest.TestCase):
    def test_overlap_is_jaccard(self):
        self.assertAlmostEqual(overlap("abc", "abc"), 1.0)
        self.assertEqual(overlap("", "abc"), 0.0)
        self.assertLess(overlap("修复登录失败", "今天天气真好"), 0.2)

    def test_repeated_tool_calls_abort(self):
        w = Watchdog(stall_steps=3)
        sig = ["read_file:login.py"]
        out = [w.observe(step=i, task="修 login", last_text="", tool_sigs=sig, history_len=i) for i in (1, 2, 3)]
        self.assertEqual([d.action for d in out[:2]], [OK, OK])
        self.assertEqual(out[2].action, ABORT)
        self.assertEqual(out[2].reason, "loop")

    def test_no_growth_abort(self):
        w = Watchdog(stall_steps=3)
        sigs = [["a"], ["b"], ["c"], ["d"]]
        out = [w.observe(step=i + 1, task="t", last_text="", tool_sigs=sigs[i], history_len=5) for i in range(4)]
        self.assertEqual(out[-1].action, ABORT)
        self.assertEqual(out[-1].reason, "no_progress")

    def test_off_topic_reminds_then_aborts(self):
        w = Watchdog(stall_steps=99, topic_floor=0.15, max_topic_trips=2, min_text_len=12)
        task = "修复 login.py 里登录失败的报错"
        off = "今天天气不错，适合出去散步看看外面的风景"
        d1 = w.observe(step=1, task=task, last_text=off, tool_sigs=["a"], history_len=10)
        self.assertEqual(d1.action, REMIND_GOAL)
        d2 = w.observe(step=2, task=task, last_text=off, tool_sigs=["b"], history_len=11)
        self.assertEqual(d2.action, ABORT)
        self.assertEqual(d2.reason, "off_topic")

    def test_on_topic_stays_ok(self):
        w = Watchdog(stall_steps=99, topic_floor=0.0)
        d = w.observe(step=1, task="修复 login.py 登录失败", last_text="正在查看 login.py 的登录逻辑", tool_sigs=["a"], history_len=3)
        self.assertEqual(d.action, OK)

    def test_short_ack_is_not_off_topic(self):
        """真实模型跑 mem-03 时「已更新为 2500 token」被误判跑偏，原因见 watchdog.anchors。"""
        w = Watchdog(stall_steps=99, topic_floor=0.15)
        d = w.observe(step=1, task="更正一下：默认上下文预算改成 2500 token。",
                      last_text="已更新为 2500 token。", tool_sigs=["a"], history_len=4)
        self.assertEqual(d.action, OK)

    def test_shared_anchor_suppresses_off_topic(self):
        """用词完全不同但命中同一个数字，说明还在处理同一件事。"""
        w = Watchdog(stall_steps=99, topic_floor=0.9, min_text_len=10)
        d = w.observe(step=1, task="把默认预算从 8000 改成 2500 并说明影响范围",
                      last_text="先确认一下影响范围：2500 会让长任务的工具输出更容易被裁掉",
                      tool_sigs=["a"], history_len=5)
        self.assertEqual(d.action, OK)

    def test_anchors_extracts_numbers_and_identifiers(self):
        # 取数字与「字母开头的 >=3 字符标识符」；.py 这类短后缀不算锚点
        self.assertEqual(anchors("改成 2500 token，见 api_client.py"), {"2500", "token", "api_client"})


class TestRouterWorker(unittest.TestCase):
    def setUp(self):
        self.cfg = make_cfg(max_workers=2, queue_factor=2)
        self.pool = WorkerPool(self.cfg, lambda: echo_agent(self.cfg))
        self.router = TaskRouter(self.cfg, self.pool, MessageFilter(self.cfg, router=None))

    def tearDown(self):
        self.pool.close()

    def test_noise_rejected_by_filter(self):
        d = self.router.route("你好")
        self.assertEqual(d.action, REJECT)
        self.assertTrue(d.reason.startswith("filter:"))
        self.assertEqual(self.router.filter.stats.by_action.get(IGNORE), 1)

    def test_new_task_gets_worker_and_counts_proactive(self):
        d = self.router.route("帮我排查 api_client.py 超时失败的问题")
        self.assertEqual(d.action, NEW_WORKER)
        self.assertTrue(d.proactive)  # 用户没要求拆分，路由主动分了工
        self.assertEqual(self.router.proactive_count, 1)

    def test_same_topic_reuses_worker_and_delivers(self):
        text = "帮我排查 api_client.py 超时失败的问题"
        d1 = self.router.route(text)
        self.assertEqual(d1.action, NEW_WORKER)
        d2 = self.router.route(text + " 补充：错误码是 504")
        self.assertEqual(d2.action, REUSE_WORKER)
        self.assertEqual(d2.worker_id, d1.worker_id)
        w = self.pool.workers[d2.worker_id]
        self.assertEqual(len(w.messages), 1)
        self.assertIn("504", w.task)  # 消息直达：拼进 Worker 自己的任务，不重建上下文
        self.assertEqual(len(self.pool.workers), 1)  # 没有多开 Worker

    def test_explicit_split_not_counted_as_proactive(self):
        d = self.router.route("请把它们拆成三个子任务并行处理，每个子任务独立跑")
        self.assertEqual(d.action, NEW_WORKER)
        self.assertFalse(d.proactive)

    def test_backpressure_rejects_only_when_queue_full(self):
        # queue_factor=2, max_workers=2 -> 最多 4 条待处理
        texts = [
            "帮我修复 login.py 的登录失败报错",
            "分析 sales.csv 里订单金额异常值",
            "排查 scheduler.py 任务堆积根因",
            "优化 api_client.py 的超时重试逻辑",
        ]
        for t in texts:
            d = self.router.route(t)
            self.assertEqual(d.action, NEW_WORKER)
        d = self.router.route("重构 report_generator.py 的分层结构")
        self.assertEqual(d.action, REJECT)
        self.assertIn("队列已满", d.reason)
        self.assertEqual(self.router.stats()["by_action"][REJECT], 1)


class TestSupervisor(unittest.TestCase):
    """两个实验臂的能量差：过滤 + 并行。"""

    ITEMS = [
        ("a", "帮我修复 login.py 里登录失败的报错"),
        ("b", "分析 sales.csv 里订单金额的异常值"),
        ("c", "排查 scheduler.py 任务堆积的根因"),
        ("n", "你好"),
    ]

    def _run(self, mode: str, latency: float = 0.03):
        cfg = make_cfg(max_workers=3, sim_latency_s=latency)
        checks = {"a": {"type": "contains", "value": "done"}}
        sup = Supervisor(cfg, lambda: echo_agent(cfg, latency_s=latency), checks=checks)
        items = [WorkItem(id=i, text=t) for i, t in self.ITEMS]
        rep = sup.run_single(items) if mode == MODE_SINGLE else sup.run_super(items)
        return rep

    def test_single_mode_runs_everything(self):
        rep = self._run(MODE_SINGLE, latency=0.0)
        self.assertEqual(rep.attempts, 4)  # 噪音也被完整跑了一遍
        self.assertEqual(rep.rejected, [])
        self.assertEqual(rep.pass_rate, 1.0)

    def test_super_mode_drops_noise(self):
        rep = self._run(MODE_SUPER, latency=0.03)
        self.assertEqual(rep.attempts, 3)
        self.assertEqual(len(rep.rejected), 1)
        self.assertEqual(rep.delegated, 3)
        self.assertEqual(len(rep.completed), 3)
        self.assertGreaterEqual(rep.peak_concurrency, 2)

    def test_super_is_faster_and_cheaper(self):
        serial = self._run(MODE_SINGLE, latency=0.04)
        parallel = self._run(MODE_SUPER, latency=0.04)
        self.assertLess(parallel.makespan_ms, serial.makespan_ms)
        self.assertLess(parallel.tokens, serial.tokens)
        self.assertTrue(serial.summary()["mode"] == MODE_SINGLE)

    def test_report_summary_shape(self):
        rep = self._run(MODE_SUPER, latency=0.0)
        s = rep.summary()
        for k in ("mode", "submitted", "processed", "completed", "rejected", "pass_rate",
                  "makespan_ms", "p95_ms", "peak_concurrency", "delegated", "proactive",
                  "tokens", "tokens_per_task", "cost_usd", "filter"):
            self.assertIn(k, s)
        self.assertAlmostEqual(s["pass_rate"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
