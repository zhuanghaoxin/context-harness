"""Agent 主循环。

设计要点全在这里，每一行都能对应一个具体问题：

  1. 目标重注入（goal_reinject_every）：长任务里模型会"跑偏"，定期把目标重新
     放回上下文尾部，比只在开头说一次有效得多（Lost in the Middle）。
  2. 重复调用检测（repeat_limit）：模型陷入 "read_file -> 同样参数 -> 再读" 死循环时，
     先注入纠偏提示，仍不止就停机，避免白烧 token。
  3. 每步预算重算：历史在长，预算不变。每步都重新 assemble 一次。
  4. 工具结果压缩：超过额度先压缩（归档可回溯），不是直接丢弃。
  5. 停机条件显式化：final / max_steps / repeated_tool / error，写进 RunResult。
"""
from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from typing import Any

from .budget import TokenBudget
from .checkpoint import CheckpointStore
from .compress import ToolOutputCompressor
from .config import Config
from .llm import Router
from .log import get_logger
from .memory import MemoryManager
from .tools import ToolRegistry, ToolSpec
from .trace import Tracer
from .types import Message, RunResult, StepRecord, ToolResult, Usage, new_id
from .tokens import estimate, estimate_messages
from .watchdog import ABORT as WD_ABORT

log = get_logger("agent")
from .watchdog import REMIND_GOAL as WD_REMIND
from .watchdog import Watchdog

SYSTEM_PROMPT = """你是一个谨慎、高效的任务执行 agent。

规则：
1. 能一次拿到信息就不要分多次调用工具；调用前先想清楚需要什么。
2. 工具报错时先读错误信息再修正参数，不要重复同样的调用。
3. 信息足够时立刻给出结论，不要为了"更完整"继续探索。
4. 回答用简洁的中文，先说结论，再给必要的依据。
5. 上下文里的【已知事实】是之前沉淀的记忆，可直接使用；与本次观察冲突时以观察为准。
"""


@dataclass
class Agent:
    router: Router
    tools: ToolRegistry
    cfg: Config
    memory: MemoryManager | None = None
    tracer: Tracer | None = None
    compressor: ToolOutputCompressor | None = None
    checkpoint: Any = None
    watchdog: Watchdog | None = None

    def __post_init__(self) -> None:
        self.tracer = self.tracer or Tracer()
        self.compressor = self.compressor or ToolOutputCompressor(self.cfg.artifact_dir)
        # 默认就要有 checkpoint：没有持久化就没有"续跑"，存算分离是空话。
        # 之前这里是「调用方不传就不建」，结果 Worker.resume 永远从第一步重来。
        if self.checkpoint is None:
            self.checkpoint = CheckpointStore(self.cfg.db_path)
        self.watchdog = self.watchdog or Watchdog(
            stall_steps=self.cfg.watchdog_stall_steps,
            topic_floor=self.cfg.watchdog_topic_floor,
            min_text_len=self.cfg.watchdog_min_text,
        )
        self.budget = TokenBudget(
            total=self.cfg.context_budget,
            tool_share=self.cfg.tool_share,
            history_share=self.cfg.history_share,
            memory_share=self.cfg.memory_share,
        )
        self._install_memory_tool()

    def _install_memory_tool(self) -> None:
        if self.memory is None or "remember_fact" in self.tools._tools:
            return

        def run(key: str, value: str) -> str:
            action = self.memory.remember_fact(key, value, source="agent")
            return f"记忆写入成功（{action}）：{key} = {value}"

        self.tools.register(
            ToolSpec(
                name="remember_fact",
                description="把一条值得长期记住的事实写入语义记忆，例如用户偏好、项目约定。",
                parameters={
                    "type": "object",
                    "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                    "required": ["key", "value"],
                },
                func=run,
            )
        )
        # 骨架自带的记忆工具不受业务白名单约束，否则它永远调不到
        if self.tools.whitelist is not None:
            self.tools.whitelist.add("remember_fact")

    # ------------------------------------------------------------------ close
    def close(self) -> None:
        """释放持久化句柄（checkpoint / memory）。幂等，可安全重复调用。

        为什么必须显式关：sqlite 连接在 Windows 上持有文件句柄，不关就删不掉库文件——
        评测重复跑、Worker 回收都会踩到这个坑（实测 PermissionError: WinError 32）。
        """
        if self.checkpoint is not None and hasattr(self.checkpoint, "close"):
            self.checkpoint.close()
        if self.memory is not None:
            self.memory.close()

    # ------------------------------------------------------------------ run
    def run(self, task: str, run_id: str | None = None, resume: bool = False,
            extra: str | None = None) -> RunResult:
        run_id = run_id or new_id("run")
        log.info("run %s start resume=%s task=%r", run_id, resume, task[:80])
        pinned = [Message(role="system", content=SYSTEM_PROMPT, pinned=True),
                  Message(role="user", content=task, pinned=True)]
        history: list[Message] = []
        step = 0
        usage = Usage()
        cost = 0.0
        stop_reason = "max_steps"
        answer = ""

        repeat_counts: dict[str, int] = {}
        loaded = False
        if resume and self.checkpoint is not None:
            snap = self.checkpoint.load(run_id)
            if snap is not None:
                loaded = True
                pinned, history = snap.pinned, snap.history
                step = snap.step
                # 账目必须一起恢复：只恢复历史会让续跑后的 usage/cost 从 0 起算，
                # 对外披露的"单任务成本"在续跑场景下就是错的。
                meta = snap.meta or {}
                u = meta.get("usage") or {}
                usage = Usage(
                    prompt_tokens=int(u.get("prompt_tokens") or 0),
                    completion_tokens=int(u.get("completion_tokens") or 0),
                )
                cost = float(meta.get("cost") or 0.0)
                repeat_counts = {str(k): int(v) for k, v in (meta.get("repeat_counts") or {}).items()}

        # 「消息直达」的落地方式：同一 Worker 收到补充消息时不重建上下文，
        # 而是把新消息接到已有 history 后面继续跑；没有历史可续时退化成合并任务。
        if extra:
            if loaded:
                history.append(Message(role="user", content=extra))
            else:
                pinned[-1] = Message(role="user", content=f"{task}\n补充：{extra}", pinned=True)
        memory_block = ""
        if self.memory is not None:
            memory_block = self.memory.recall_block(task, budget_tokens=int(self.cfg.context_budget * 0.15))

        while step < self.cfg.max_steps:
            step += 1

            # 目标重注入：把目标贴回上下文尾部，抵抗长上下文注意力衰减
            if step > 1 and (step - 1) % self.cfg.goal_reinject_every == 0:
                history.append(
                    Message(
                        role="user",
                        content=f"【目标提醒】你的目标仍然是：{task}\n当前第 {step} 步，请判断是否已可以收敛。",
                    )
                )

            messages, budget_report = self.budget.assemble(pinned, memory_block, history)

            try:
                resp = self.router.complete(messages, self.tools.schemas())
            except Exception as e:
                self.tracer.note(f"LLM 调用失败：{type(e).__name__}: {e}")
                stop_reason = "error"
                answer = f"LLM 调用失败：{e}"
                break

            usage = usage + resp.usage
            cost += resp.cost_usd
            tool_names = [c.name for c in resp.message.tool_calls]
            self.tracer.add(
                StepRecord(
                    step=step,
                    model=resp.model,
                    prompt_tokens=resp.usage.prompt_tokens,
                    completion_tokens=resp.usage.completion_tokens,
                    cost_usd=resp.cost_usd,
                    latency_ms=resp.latency_ms,
                    tools=tool_names,
                    note=f"budget {budget_report.tokens_after}/{budget_report.budget}"
                    f" dropped={budget_report.dropped_groups}",
                )
            )

            if not resp.message.tool_calls:
                answer = resp.message.content.strip()
                stop_reason = "final"
                history.append(Message(role="assistant", content=answer))
                break

            # 执行工具（串行；并行执行见 README 的 roadmap）
            history.append(Message(role="assistant", content=resp.message.content, tool_calls=resp.message.tool_calls))
            step_sigs: list[str] = []
            for idx, call in enumerate(resp.message.tool_calls):
                # 重复调用签名（也用作工具结果的幂等键的一部分）
                sig = f"{call.name}:{json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)}"
                replay_key = f"{idx}:{sig}"
                cached = None
                if self.checkpoint is not None:
                    cached = self.checkpoint.load_tool_result(run_id, step, replay_key)

                if cached is not None:
                    # 续跑回放：这一步的工具副作用在执行时已经发生过，
                    # 重跑不能再来一次（写文件/发消息不幂等），直接用当时的结果续上。
                    result = ToolResult(
                        call.id,
                        call.name,
                        bool(cached.get("ok")),
                        output=cached.get("output") or "",
                        error=cached.get("error"),
                        omitted_chars=int(cached.get("omitted_chars") or 0),
                        ref=cached.get("ref"),
                    )
                    self.tracer.note(f"replay tool result: {call.name} (step {step})")
                else:
                    result = self.tools.call(call.name, call.arguments, call.id)

                repeat_counts[sig] = repeat_counts.get(sig, 0) + 1
                if repeat_counts[sig] == self.cfg.repeat_limit:
                    result.output = (result.output or "") + (
                        f"\n[警告] 你已经用完全相同的参数调用 {call.name} {repeat_counts[sig]} 次，"
                        "结果不会变。请换策略或直接给出结论。"
                    )
                elif repeat_counts[sig] > self.cfg.repeat_limit:
                    stop_reason = "repeated_tool"
                    answer = f"检测到重复调用 {call.name}，已停机以避免无效消耗。"
                    self.tracer.note(f"repeated tool call: {sig}")
                    return self._finish(
                        run_id, task, answer, stop_reason, step, usage, cost, history,
                        pinned=pinned, repeat_counts=repeat_counts,
                    )

                # 工具输出压缩（超额度才动）：压缩后的文本才是续跑回放的内容，
                # 否则回放会重新压缩、结果与首次执行不一致。
                if result.ok and estimate(result.output) > 600:
                    head_tokens = estimate_messages(
                        pinned + [Message(role="system", content=memory_block)] + history
                    )
                    quota = self.budget.available_for_tool_output(head_tokens, 0)
                    squeezed = self.compressor.compress(result.call_id, result.name, result.output, quota)
                    result.output = squeezed.text
                    result.omitted_chars = squeezed.omitted_chars
                    result.ref = squeezed.ref

                if cached is None and self.checkpoint is not None:
                    self.checkpoint.save_tool_result(
                        run_id, step, replay_key,
                        {
                            "ok": result.ok,
                            "output": result.output,
                            "error": result.error,
                            "omitted_chars": result.omitted_chars,
                            "ref": result.ref,
                        },
                    )

                history.append(result.to_message())
                # Watchdog 的"停滞"签名要包含**输出**：同一个调用失败一次后重试、
                # 或者读不同文件，都不算原地打转。只看工具名会把正常重试误杀。
                step_sigs.append(
                    f"{sig}|{zlib.crc32((result.output or '').encode('utf-8'))}"
                )

            # Watchdog：每一步都问一次"还要不要继续"。确定性判断，不调模型。
            directive = self.watchdog.observe(
                step=step,
                task=task,
                last_text=resp.message.content,
                tool_sigs=step_sigs,
                history_len=len(history),
            )
            if directive.action == WD_ABORT:
                stop_reason = f"watchdog:{directive.reason}"
                answer = f"Watchdog 停机（{directive.reason}）：{directive.detail}"
                self.tracer.note(f"watchdog abort: {directive.detail}")
                return self._finish(
                    run_id, task, answer, stop_reason, step, usage, cost, history,
                    pinned=pinned, repeat_counts=repeat_counts,
                )
            if directive.action == WD_REMIND:
                history.append(
                    Message(
                        role="user",
                        content=f"【目标提醒·跑偏】{directive.detail}\n原始目标：{task}",
                    )
                )

            if self.checkpoint is not None:
                self.checkpoint.save(
                    run_id, step, task, pinned, history,
                    meta={
                        "usage": usage.to_dict(),
                        "cost": cost,
                        "repeat_counts": repeat_counts,
                        "stop_reason": stop_reason,
                    },
                )

        return self._finish(
            run_id, task, answer, stop_reason, step, usage, cost, history,
            pinned=pinned, repeat_counts=repeat_counts,
        )

    def _finish(
        self,
        run_id: str,
        task: str,
        answer: str,
        stop_reason: str,
        step: int,
        usage: Usage,
        cost: float,
        history: list[Message],
        pinned: list[Message] | None = None,
        repeat_counts: dict[str, int] | None = None,
    ) -> RunResult:
        # 早退路径（repeated_tool / watchdog / final / error）也要落盘：
        # 不保存的话，续跑会从更早的 step 重来，已经付过的 token 和已发生的副作用都对不上账。
        if self.checkpoint is not None and pinned is not None:
            self.checkpoint.save(
                run_id, step, task, pinned, history,
                meta={
                    "usage": usage.to_dict(),
                    "cost": cost,
                    "repeat_counts": repeat_counts or {},
                    "stop_reason": stop_reason,
                },
            )
        # 把本次 episode 写入情景记忆，供后续同名任务召回
        if self.memory is not None and answer:
            self.memory.record_episode(
                f"任务：{task[:200]}\n结论：{answer[:400]}",
                meta={"run_id": run_id, "stop_reason": stop_reason},
            )
        log.info(
            "run %s stop=%s steps=%d prompt_tokens=%d completion_tokens=%d cost=%.6f",
            run_id, stop_reason, step, usage.prompt_tokens, usage.completion_tokens, cost,
        )
        return RunResult(
            answer=answer,
            stop_reason=stop_reason,
            steps=step,
            usage=usage,
            cost_usd=cost,
            trace=list(self.tracer.records),
            run_id=run_id,
        )
