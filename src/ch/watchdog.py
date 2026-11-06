"""Watchdog：检测"停滞"和"跑偏"，并给出纠偏动作。

要回答的问题是"Agent 卡住了怎么办"。三种卡法要分开处理，因为修复手段不同：

  loop        同一组工具调用原地打转          -> 直接停机（重试同样的调用不会变好）
  no_progress 步数在涨，但没有新信息进来      -> 提醒目标 + 收缩预算
  off_topic   内容离原始目标越来越远          -> 重注入目标；连续两次才停机

设计原则：Watchdog 只做**确定性判断**，不调模型。原因有两个：
  1. 它本身是成本控制器，再用大模型监控会反过来烧钱；
  2. 出问题的时候最可疑的就是模型，拿模型去监控模型没有独立性。

与 Agent 的分工：Agent 负责"这一步做什么"，Watchdog 只回答"还要不要继续"。
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any

OK = "ok"
REMIND_GOAL = "remind_goal"
ABORT = "abort"

_WORD_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+", re.I)


@dataclass
class Directive:
    action: str = OK
    reason: str = ""
    detail: str = ""

    @property
    def abort(self) -> bool:
        return self.action == ABORT

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "reason": self.reason, "detail": self.detail}


def _fingerprint(text: str) -> set[str]:
    """字符 2-gram + 英文词，用于判断"是否还在同一个话题上"。"""
    s = (text or "").strip().lower()
    if not s:
        return set()
    grams: set[str] = set()
    grams.update(w for w in _WORD_RE.findall(s) if len(w) >= 2)
    for i in range(len(s) - 1):
        pair = s[i : i + 2]
        if pair.strip():
            grams.add(pair)
    return grams


def overlap(a: str, b: str) -> float:
    """Jaccard 相似度，[0, 1]。零依赖、对中文友好、结果稳定可测。"""
    fa, fb = _fingerprint(a), _fingerprint(b)
    if not fa or not fb:
        return 0.0
    return len(fa & fb) / len(fa | fb)


_IDENT_RE = re.compile(r"\d+(?:\.\d+)?|[a-zA-Z][a-zA-Z0-9_]{2,}")


def anchors(text: str) -> set[str]:
    """数字与标识符：任务里出现的 2500、8000、api_client.py 这类"硬锚点"。

    一段短文本只要命中了锚点就说明它还在处理这件事，哪怕用词完全不同。
    真实模型跑 mem-03（"预算改成 2500 token"）时，回复"已更新为 2500 token"，
    与问题的 Jaccard 只有 0.02 —— 纯词面相似度会把这种正确的确认判成跑偏。
    """
    return {m.lower() for m in _IDENT_RE.findall(text or "")}


@dataclass
class WatchdogStats:
    checks: int = 0
    loop_trips: int = 0
    stall_trips: int = 0
    topic_trips: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "checks": self.checks,
            "loop_trips": self.loop_trips,
            "stall_trips": self.stall_trips,
            "topic_trips": self.topic_trips,
        }


class Watchdog:
    def __init__(
        self,
        stall_steps: int = 3,
        topic_floor: float = 0.15,
        max_topic_trips: int = 2,
        min_text_len: int = 40,
    ):
        self.stall_steps = max(2, stall_steps)
        self.topic_floor = topic_floor
        self.max_topic_trips = max_topic_trips
        self.min_text_len = min_text_len

        self.stats = WatchdogStats()
        self._sig_window: deque[str] = deque(maxlen=self.stall_steps)
        self._last_growth: int | None = None
        self._no_growth = 0
        self._topic_trips = 0
        self._last_step = 0

    # ------------------------------------------------------------- observe
    def observe(
        self,
        step: int,
        task: str,
        last_text: str = "",
        tool_sigs: list[str] | None = None,
        history_len: int = 0,
    ) -> Directive:
        """每步调用一次。step 回退视为新一轮任务（DAG 复用同一个 watchdog 时）。"""
        if step <= self._last_step:
            self._reset_window()
        self._last_step = step
        self.stats.checks += 1

        sig = "|".join(tool_sigs or [])

        # 1) 完全相同的工具签名连续出现 -> 死循环
        self._sig_window.append(sig)
        if (
            len(self._sig_window) == self.stall_steps
            and sig
            and len(set(self._sig_window)) == 1
        ):
            self.stats.loop_trips += 1
            return Directive(ABORT, "loop", f"连续 {self.stall_steps} 步工具调用完全相同：{sig[:80]}")

        # 2) 历史没有增长 -> 没有新信息进来
        if self._last_growth is not None and history_len <= self._last_growth:
            self._no_growth += 1
        else:
            self._no_growth = 0
        self._last_growth = history_len
        if self._no_growth >= self.stall_steps:
            self.stats.stall_trips += 1
            return Directive(
                ABORT, "no_progress", f"连续 {self._no_growth} 步上下文没有新增内容"
            )

        # 3) 内容跑偏：只对"够长的成篇输出"判，且必须连硬锚点都没命中。
        #    两个排除条件都是在真实模型上撞出来的：
        #    - 短文本（"已更新为 2500 token"）只是确认，词面 Jaccard 天然很低；
        #    - 命中任务里的数字/标识符，说明还在处理同一件事，用词不同不算跑偏。
        text = (last_text or "").strip()
        if len(text) >= self.min_text_len and (task or "").strip():
            shared = anchors(text) & anchors(task)
            score = overlap(text, task)
            if score < self.topic_floor and not shared:
                self._topic_trips += 1
                self.stats.topic_trips += 1
                if self._topic_trips >= self.max_topic_trips:
                    return Directive(
                        ABORT, "off_topic", f"与目标重合度 {score:.3f} < {self.topic_floor}，已连续 {self._topic_trips} 次"
                    )
                return Directive(
                    REMIND_GOAL,
                    "off_topic",
                    f"与目标重合度 {score:.3f} 偏低，重注入目标",
                )
            self._topic_trips = 0

        return Directive(OK)

    def _reset_window(self) -> None:
        self._sig_window.clear()
        self._last_growth = None
        self._no_growth = 0
        self._topic_trips = 0
