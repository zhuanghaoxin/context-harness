"""两级消息过滤：Super-agent 的第一道工程闸门。

为什么"主动找活"不能靠提示词：

    DeepSeek Super-agent 的对照实验里，中性提示词下连跑五轮，**一次都没有**主动
    调用子 Agent；把"鼓励委派"写进提示词之后，五轮里有四轮用了。这说明
    "让 Agent 自己判断该不该接手"必须是**工程机制**，提示词只能锦上添花。

两级设计（对应论文里"程序规则 + 大模型语义判断"的组合）：

    L1 规则层（零成本）：正则、长度、任务动词、错误关键词。把明显的噪音和明显的
       任务直接判掉。绝大多数消息在这一层就结束了，不花一分钱。
    L2 模型层：只有落在 (low, high) 灰区的消息才交给小模型四分类。

关键取舍：**灰区才调模型**。过滤器的调用次数本身就是要优化的指标，最后落在
    FilterStats 里——"过滤器成本占多少"是必然要回答的下一层问题。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

ACT_NOW = "act_now"      # 值得立刻处理
DEFER = "defer"          # 有价值但上下文不足，暂缓
IGNORE = "ignore"        # 噪音，永不处理
CLARIFY = "clarify"      # 意图不明，需要反问

_ACTIONS = (ACT_NOW, DEFER, IGNORE, CLARIFY)

# ---------------------------------------------------------------- L1 规则
_NOISE_PATTERNS = [
    re.compile(r"^\s*$"),
    re.compile(
        r"^(ok|okay|k|收到|好的|好|嗯+|哈+|嘿+|hello|hi|在吗|在不在|早|晚安|你好|您好|谢谢|辛苦了|多谢)[!！。.,，\s~～]*$",
        re.I,
    ),
    re.compile(r"^(ping|pong|heartbeat|health|健康检查|测试消息)$", re.I),
    re.compile(r"^[\W_]{1,6}$"),                                  # 纯标点 / 表情
    re.compile(r"^\s*\d{4}-\d\d-\d\d[T ]\d\d:\d\d"),              # 日志时间行
    re.compile(r"^(GET|POST|PUT|DELETE|PATCH)\s+/\S*", re.I),     # 访问日志
    re.compile(r"^\s*(at\s+\S+\(|Traceback)", re.I),              # 栈帧
]

_TASK_HINTS = (
    "帮我", "请", "麻烦", "需要", "要", "实现", "修复", "修一下", "写一个", "写个",
    "整理", "分析", "对比", "生成", "检查", "定位", "复现", "排查", "重构", "优化",
    "为什么", "如何", "怎么", "能不能", "是否",
)
_INTENT_WORDS = (
    "bug", "报错", "失败", "异常", "错误", "traceback", "error", "panic", "超时",
    "很慢", "变慢", "性能", "内存", "泄漏", "崩溃", "回归", "pr", "commit",
    "测试", "单测", "部署", "回滚", "上线", "覆盖率",
)
_CODE_HINTS = (
    "def ", "class ", "function ", "```", "()", "==", "->", "=>",
    ".py", ".ts", ".js", ".go", ".java", ".md", ".json",
)
_PATH_RE = re.compile(r"[\w.-]+/[\w./-]+|[A-Za-z]:\\[\w\\ .-]+")


@dataclass
class RuleResult:
    score: float
    reasons: list[str] = field(default_factory=list)
    noise: bool = False


def rule_score(text: str) -> RuleResult:
    """零成本的启发式打分，值域 [0, 1]。这是 L1 层的全部实现。"""
    s = (text or "").strip()
    if not s:
        return RuleResult(0.0, ["empty"], noise=True)

    for pat in _NOISE_PATTERNS:
        if pat.search(s):
            return RuleResult(0.05, [f"noise:{pat.pattern[:24]}"], noise=True)

    score = 0.0
    reasons: list[str] = []

    n = len(s)
    if n >= 12:
        score += 0.20
        reasons.append("len>=12")
    if n >= 40:
        score += 0.15
        reasons.append("len>=40")
    if n < 6:
        score -= 0.15
        reasons.append("len<6")

    low = s.lower()
    if any(h in s or h in low for h in _TASK_HINTS):
        score += 0.25
        reasons.append("task_hint")
    if any(w in low for w in _INTENT_WORDS):
        score += 0.20
        reasons.append("intent")
    if any(c in s for c in _CODE_HINTS) or _PATH_RE.search(s):
        score += 0.15
        reasons.append("code_or_path")
    if any(q in s for q in ("?", "？", "吗")):
        score += 0.10
        reasons.append("question")

    return RuleResult(max(0.0, min(1.0, score)), reasons)


# ---------------------------------------------------------------- 决策
@dataclass
class FilterDecision:
    action: str
    score: float
    stage: str          # rule | model
    reason: str = ""

    @property
    def accepted(self) -> bool:
        return self.action in (ACT_NOW, CLARIFY)

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "score": round(self.score, 3), "stage": self.stage, "reason": self.reason}


@dataclass
class FilterStats:
    total: int = 0
    rule_act: int = 0
    rule_drop: int = 0
    model_called: int = 0
    model_failed: int = 0
    by_action: dict[str, int] = field(default_factory=dict)

    @property
    def model_ratio(self) -> float:
        return self.model_called / self.total if self.total else 0.0

    def bump(self, action: str) -> None:
        self.by_action[action] = self.by_action.get(action, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "rule_act": self.rule_act,
            "rule_drop": self.rule_drop,
            "model_called": self.model_called,
            "model_failed": self.model_failed,
            "model_ratio": round(self.model_ratio, 3),
            "by_action": dict(self.by_action),
        }


_FILTER_PROMPT = """你是消息过滤器。判断这条消息对"任务执行 agent"是否有价值。

只输出一行，格式严格为：ACTION: <act_now|defer|ignore|clarify>

- act_now : 明确的任务、问题、报错、需求，现在就该处理
- defer   : 有信息量但现在没有可执行动作（闲聊延伸、背景补充）
- ignore  : 噪音（打招呼、心跳、日志、重复确认）
- clarify : 意图不明确，需要先反问

不要解释，不要输出其它内容。"""


class MessageFilter:
    """两级过滤器。router 为 None 时退化为纯规则（离线/单测模式）。"""

    def __init__(self, cfg: Any = None, router: Any = None):
        self.cfg = cfg
        self.router = router
        self.stats = FilterStats()
        self.high = getattr(cfg, "filter_high", 0.70)
        self.low = getattr(cfg, "filter_low", 0.30)
        self.enabled = getattr(cfg, "filter_enabled", True)

    def decide(self, text: str) -> FilterDecision:
        if not self.enabled:
            self.stats.total += 1
            self.stats.bump(ACT_NOW)
            return FilterDecision(ACT_NOW, 1.0, "off", "filter disabled")

        self.stats.total += 1
        r = rule_score(text)

        # 明显噪音：直接丢，不调模型
        if r.noise:
            self.stats.rule_drop += 1
            self.stats.bump(IGNORE)
            return FilterDecision(IGNORE, r.score, "rule", ",".join(r.reasons))

        # 明显任务：直接放行，不调模型（省钱的关键一步）
        if r.score >= self.high:
            self.stats.rule_act += 1
            self.stats.bump(ACT_NOW)
            return FilterDecision(ACT_NOW, r.score, "rule", ",".join(r.reasons))

        # 明显无价值：不调模型
        if r.score <= self.low:
            self.stats.rule_drop += 1
            action = DEFER if r.score > 0.1 else IGNORE
            self.stats.bump(action)
            return FilterDecision(action, r.score, "rule", ",".join(r.reasons))

        # ---- 灰区：只有这里才花 utility 模型的调用
        action = self._ask_model(text, r.score)
        self.stats.bump(action)
        return FilterDecision(action, r.score, "model", ",".join(r.reasons))

    def _ask_model(self, text: str, fallback_score: float) -> str:
        # 灰区默认 fail-open：漏掉一条真实任务的代价，远大于多处理一条噪音。
        # 明确的噪音在 L1 规则层已经拦掉了，走到这里说明规则层认不出来。
        if self.router is None:
            return ACT_NOW

        self.stats.model_called += 1
        try:
            resp = self.router.complete(
                [_sys(_FILTER_PROMPT), _user(text[:1200])], None, role="utility"
            )
            action = _parse_action(getattr(resp.message, "content", "") or "")
            if action:
                return action
        except Exception:
            self.stats.model_failed += 1

        # 模型没给出可用答案（离线 ScriptedLLM / 报错）：同样 fail-open
        return ACT_NOW


def _sys(text: str) -> Any:
    from .types import Message

    return Message(role="system", content=text)


def _user(text: str) -> Any:
    from .types import Message

    return Message(role="user", content=text)


def _parse_action(content: str) -> str:
    m = re.search(r"ACTION:\s*([a-z_]+)", content, re.I)
    if not m:
        return ""
    a = m.group(1).lower()
    return a if a in _ACTIONS else ""
