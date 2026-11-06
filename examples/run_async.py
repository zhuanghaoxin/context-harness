"""v2.1 演示：Worker 异步执行 + 存算分离 + 凭 id 续跑。

    python examples/run_async.py

四段：
  1) 提交即返回 worker_id，主线程只拿 id，不等结果
  2) 只凭 id 从库里读状态（这个进程不再持有 Worker 对象）
  3) 模拟进程被杀：被截断的 run 在库里留了 interrupted/running 痕迹
  4) 凭同一个 id 从 checkpoint 续跑，收敛出最终答案

离线可跑：用按调用次数应答的脚本模型，不联网、不需要 key。
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ch.agent import Agent  # noqa: E402
from ch.config import Config  # noqa: E402
from ch.llm import Router, ScriptedLLM  # noqa: E402
from ch.router_worker import WorkerPool  # noqa: E402
from ch.tools import default_registry  # noqa: E402
from ch.worker_store import WorkerStore  # noqa: E402

ART = ROOT / ".artifacts" / "async"


def make_cfg(max_steps: int) -> Config:
    cfg = Config.from_env()
    cfg.base_url = cfg.api_key = cfg.model = ""
    cfg.context_budget = 2000
    cfg.max_steps = max_steps
    cfg.db_path = str(ART / "checkpoints.sqlite3")
    cfg.artifact_dir = str(ART)
    cfg.ensure_dirs()
    return cfg


def make_factory(cfg: Config):
    """三步才收敛：查工具 -> 再查一次 -> 给结论。用步数换"能被打断"的空间。"""

    def factory() -> Agent:
        llm = ScriptedLLM(
            [
                {"tool": "calculator", "args": {"expression": "1200*1.05"}},
                {"tool": "calculator", "args": {"expression": "1200*1.05+800"}},
                {"text": "结论：含税总价 2060 元，运费 800 元已包含在内。"},
            ],
            model_name="demo-llm",
            latency_s=0.08,
            price=(0.15, 0.60),
        )
        return Agent(
            router=Router(cfg, main=llm, utility=llm),
            tools=default_registry(root=str(ROOT), approval=lambda n, a: False, whitelist={"calculator"}),
            cfg=cfg,
            memory=None,
        )

    return factory


def main() -> None:
    if ART.exists():
        shutil.rmtree(ART, ignore_errors=True)
    store_path = ART / "workers.sqlite3"

    # ---- 1) 异步提交
    cfg = make_cfg(max_steps=1)  # 故意只给 1 步：第一次必然被截断
    pool = WorkerPool(cfg, make_factory(cfg), max_workers=2, store=WorkerStore(store_path))
    t0 = time.time()
    ids = [pool.submit(f"算一下订单 {i} 的含税总价") for i in range(3)]
    print("=" * 74)
    print("1) 异步提交：submit() 立刻返回 id，不等结果")
    print("=" * 74)
    print(f"  提交 3 条耗时：{(time.time() - t0) * 1000:.1f} ms（不阻塞）")
    print("  worker_id：", ids)
    print("  刚提交时的状态：", [pool.poll(i)["status"] for i in ids])

    for i in ids:
        pool.wait(i, timeout=15)
    print("  跑完后的状态：  ", [pool.poll(i)["status"] for i in ids])
    print("  stop_reason：   ", [pool.poll(i)["stop_reason"] for i in ids], "<- 被 max_steps 截断")
    pool.close()

    # ---- 2) 只凭 id 读状态
    print()
    print("=" * 74)
    print("2) 存算分离：主进程已经不再持有任何 Worker 对象")
    print("=" * 74)
    other = WorkerStore(store_path)  # 换一个连接，模拟另一个进程
    rec = other.load(ids[0])
    print("  另一个进程只凭 id 读到的记录：")
    print("   ", rec.to_dict() if rec else None)
    interrupted = other.mark_interrupted()
    print(f"  启动时把残留的 running 标成 interrupted：{interrupted} 条（本例已跑完，故为 0）")

    # ---- 3) 续跑
    print()
    print("=" * 74)
    print("3) 凭同一个 id 从 checkpoint 续跑（这次给足 6 步）")
    print("=" * 74)
    long_cfg = make_cfg(max_steps=6)
    pool2 = WorkerPool(long_cfg, make_factory(long_cfg), max_workers=2, store=WorkerStore(store_path))
    for i in ids:
        pool2.resume(i)
    for i in ids:
        pool2.wait(i, timeout=20)
    for i in ids:
        r = pool2.poll(i)
        print(f"  {i}: {r['status']:<6} steps={r['steps']} stop={r['stop_reason']}")
        print(f"      answer={r['answer']}")
    pool2.close()
    other.close()

    print()
    print("要点：")
    print("  1) 续跑能成立，是因为 Agent 每步把 pinned/history/step 落在 checkpoint 里，")
    print("     且 run_id 直接复用 worker_id —— 不需要在进程间传任何执行状态。")
    print("  2) 状态库只存元信息（任务文本、状态、用量），执行进度归 checkpoint，职责不重叠。")
    print("  3) 真挂掉时，靠 mark_interrupted() 把上次残留的 running 找出来再续跑。")


if __name__ == "__main__":
    main()
