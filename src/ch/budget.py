"""Token 预算分配 + 上下文裁剪。

这是整个 harness 的核心机制所在，也是最容易出问题的设计点：
"上下文快满了你先砍什么？"

本实现的答案（可解释、可复现）：
  - 固定区（pinned）：system prompt + 原始目标，永不裁剪；
  - 记忆区：按 memory_share 配额，超了先截断记忆文本（信息密度最低）；
  - 历史区：按 history_share 配额，从**最旧**开始整组丢弃；
  - 工具区：按 tool_share 配额，单次工具输出超限按 token 精确硬截断。

token ↔ 字符不是线性关系（CJK 1 字≈1 token，ASCII 4 字符≈1 token），所以裁剪
一律用 estimate() 反推字符数（二分收敛），不能用固定字符比例。

配对完整性：丢弃一个带 tool_calls 的 assistant 消息时，必须同时丢弃它对应的
tool 结果消息，否则 API 会报 "tool_call_id not found"。这里用 group 保证。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .types import Message
from .tokens import estimate, estimate_messages

MEMORY_MARKER = "\n...[memory truncated]"
HARD_TRUNCATE_MARKER = "\n...[hard truncated by budget]"


# --------------------------------------------------------------- token 裁剪
def fit_prefix_chars(text: str, target_tokens: int) -> int:
    """最长前缀字符数 n，使 estimate(text[:n]) <= target_tokens。

    estimate() 对前缀单调不减，所以直接二分，不需要按固定字符比例猜。
    """
    if target_tokens <= 0 or not text:
        return 0
    if estimate(text) <= target_tokens:
        return len(text)
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate(text[:mid]) <= target_tokens:
            lo = mid
        else:
            hi = mid - 1
    return lo


def fit_suffix_chars(text: str, target_tokens: int) -> int:
    """最长后缀字符数 n，使 estimate(text[-n:]) <= target_tokens。"""
    if target_tokens <= 0 or not text:
        return 0
    if estimate(text) <= target_tokens:
        return len(text)
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate(text[len(text) - mid :]) <= target_tokens:
            lo = mid
        else:
            hi = mid - 1
    return lo


def truncate_to_tokens(text: str, target_tokens: int, marker: str = "") -> str:
    """按 token 预算裁剪 text，保证 estimate(结果) <= target_tokens。

    marker 的 token 开销会先被预留，所以中文/英文/混排都按真实估算收敛，
    不会出现"截了 300 字符仍然超预算"的情况。
    """
    if not text:
        return text
    if target_tokens <= 0:
        return ""
    if estimate(text) <= target_tokens:
        return text
    body = target_tokens - estimate(marker)
    if body <= 0:
        # marker 本身就放不下：退化为纯裁剪，优先保证不超预算
        return text[: fit_prefix_chars(text, target_tokens)]
    return text[: fit_prefix_chars(text, body)] + marker


# --------------------------------------------------------------- 报告
@dataclass
class BudgetReport:
    tokens_before: int = 0
    tokens_after: int = 0
    budget: int = 0
    dropped_groups: int = 0
    dropped_messages: list[str] = field(default_factory=list)
    memory_truncated: bool = False
    hard_truncated: bool = False
    history_quota: int = 0
    tool_quota: int = 0
    history_trimmed: bool = False

    def to_dict(self) -> dict:
        return {
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "budget": self.budget,
            "dropped_groups": self.dropped_groups,
            "dropped_messages": self.dropped_messages,
            "memory_truncated": self.memory_truncated,
            "hard_truncated": self.hard_truncated,
            "history_quota": self.history_quota,
            "tool_quota": self.tool_quota,
            "history_trimmed": self.history_trimmed,
        }


def build_groups(history: list[Message]) -> list[list[Message]]:
    """把历史切成"不可拆分"的组：(assistant + 它的 tool 结果) 为一组。"""
    groups: list[list[Message]] = []
    i = 0
    n = len(history)
    while i < n:
        m = history[i]
        group = [m]
        i += 1
        if m.role == "assistant" and m.tool_calls:
            ids = {c.id for c in m.tool_calls}
            while i < n and history[i].role == "tool" and history[i].tool_call_id in ids:
                group.append(history[i])
                i += 1
        groups.append(group)
    return groups


class TokenBudget:
    def __init__(
        self,
        total: int = 8000,
        tool_share: float = 0.45,
        history_share: float = 0.35,
        memory_share: float = 0.20,
    ):
        self.total = total
        self.tool_share = tool_share
        self.history_share = history_share
        self.memory_share = memory_share

    def assemble(
        self, pinned: list[Message], memory_block: str, history: list[Message]
    ) -> tuple[list[Message], BudgetReport]:
        rep = BudgetReport(budget=self.total)
        rep.tokens_before = estimate_messages(pinned) + estimate(memory_block) + estimate_messages(history)

        pinned_tokens = estimate_messages(pinned)
        leftover = max(0, self.total - pinned_tokens)

        # --- 记忆区：leftover * memory_share ---
        mem_quota = int(leftover * self.memory_share)
        memory_block = memory_block or ""
        if memory_block and estimate(memory_block) > mem_quota:
            memory_block = truncate_to_tokens(memory_block, mem_quota, marker=MEMORY_MARKER)
            rep.memory_truncated = True

        remaining = leftover - estimate(memory_block)

        # --- 历史区 / 工具区：各自的独立配额 ---
        groups = build_groups([m for m in history if not m.pinned])
        history_quota = int(remaining * self.history_share)
        tool_quota = int(remaining * self.tool_share)
        total_quota = remaining
        rep.history_quota = history_quota
        rep.tool_quota = tool_quota

        # 1) 总配额兜底：整组丢最旧的（保留最后一组 = 最近状态）
        while len(groups) > 1 and estimate_messages(_flat(groups)) > total_quota:
            _drop_oldest(groups, rep)

        # 2) 历史区配额：非 tool 消息超限，同样丢最旧
        while len(groups) > 1 and _class_tokens(groups, "tool") + _history_tokens(groups) > (
            history_quota + tool_quota
        ) and _history_tokens(groups) > history_quota:
            _drop_oldest(groups, rep)
            rep.history_trimmed = True

        # 3) 工具区配额：tool 结果超限，按 token 精确硬截断（先砍最大的一条）
        overflow = _class_tokens(groups, "tool") - tool_quota
        if overflow > 0:
            overflow, trimmed = _shrink_tool_outputs(groups, overflow)
            rep.hard_truncated = rep.hard_truncated or trimmed > 0

        # 4) 总配额再兜底：单组过大时仍可能超总预算
        used = estimate_messages(_flat(groups))
        if used > total_quota:
            overflow = used - total_quota
            _, trimmed = _shrink_tool_outputs(groups, overflow)
            rep.hard_truncated = rep.hard_truncated or trimmed > 0
            while len(groups) > 1 and estimate_messages(_flat(groups)) > total_quota:
                _drop_oldest(groups, rep)

        mem_msg = [Message(role="system", content=memory_block, name="recalled_memory")] if memory_block else []
        out = list(pinned) + mem_msg + _flat(groups)

        # 清理孤立的 tool 消息（防御性）
        out = _drop_orphan_tools(out)

        rep.tokens_after = estimate_messages(out)
        return out, rep

    def available_for_tool_output(self, pinned_tokens: int, memory_tokens: int) -> int:
        leftover = max(0, self.total - pinned_tokens - memory_tokens)
        return int(leftover * self.tool_share)


# --------------------------------------------------------------- 内部工具
def _flat(groups: list[list[Message]]) -> list[Message]:
    return [m for g in groups for m in g]


def _brief(g: list[Message]) -> str:
    if not g:
        return "(empty group)"
    m = g[0]
    return f"{m.role}:{(m.content or '')[:40]}"


def _history_tokens(groups: list[list[Message]]) -> int:
    """历史区用量：非 tool 消息（user/assistant，含 tool_calls 结构开销）。"""
    return estimate_messages([m for m in _flat(groups) if m.role != "tool"])


def _class_tokens(groups: list[list[Message]], cls: str) -> int:
    if cls == "tool":
        return estimate_messages([m for m in _flat(groups) if m.role == "tool"])
    return _history_tokens(groups)


def _drop_oldest(groups: list[list[Message]], rep: BudgetReport) -> None:
    dropped = groups.pop(0)
    rep.dropped_groups += 1
    rep.dropped_messages.append(_brief(dropped))


def _shrink_tool_outputs(groups: list[list[Message]], overflow: int) -> tuple[int, int]:
    """按 token 预算裁剪 tool 输出，返回 (剩余 overflow, 被裁剪的消息数)。

    每条 tool 结果至少保留 64 token 的头部，避免裁成空串让模型彻底失去信息。
    """
    tool_msgs = [m for m in _flat(groups) if m.role == "tool"]
    tool_msgs.sort(key=lambda m: estimate(m.content), reverse=True)
    trimmed = 0
    for m in tool_msgs:
        if overflow <= 0:
            break
        cur = estimate(m.content)
        if cur <= 64:
            continue
        cut = min(cur - 64, overflow)
        m.content = truncate_to_tokens(m.content, cur - cut, marker=HARD_TRUNCATE_MARKER)
        m.meta["hard_truncated"] = True
        trimmed += 1
        overflow -= cut
    return overflow, trimmed


def _drop_orphan_tools(messages: list[Message]) -> list[Message]:
    declared = {c.id for m in messages for c in m.tool_calls}
    return [m for m in messages if not (m.role == "tool" and m.tool_call_id not in declared)]
