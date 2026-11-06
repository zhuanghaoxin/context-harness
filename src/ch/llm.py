"""LLM 客户端：OpenAI 兼容协议（urllib 实现，零依赖）+ 离线 Mock + 混合路由。

为什么自己写而不用 openai SDK：
  1. 只依赖 /chat/completions 一个端点，任何兼容服务都能接；
  2. 能讲清"一次调用到底发生了什么"，而不是把 SDK 当黑盒。

混合路由（成本优化的关键手段之一）：
  - main    : 主推理模型，负责规划与决策；
  - utility : 小模型，负责工具输出摘要、参数修正、意图分类。
  这两类调用的 token 结构完全不同（utility 是"长输入短输出"），
  用小模型承接可以显著压低单任务成本，见 evaluate.py 的对照实验。
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Protocol, Sequence

from .config import Config
from .types import LLMResponse, Message, ToolCall, Usage


class LLM(Protocol):
    model_name: str

    def complete(self, messages: Sequence[Message], tools: list[dict] | None = None) -> LLMResponse:
        ...


class LLMError(RuntimeError):
    """LLM 调用失败。retryable 标记该错误是否值得退避重试。"""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


# 429 限流、408/409/425 与 5xx 属于"稍后重试可能成功"；其余 4xx 是请求本身有问题，
# 重试只会浪费时间与额度（真实模型按 token 计费）。
_RETRYABLE_HTTP = {408, 409, 425, 429}


def _retryable_http(code: int) -> bool:
    return code in _RETRYABLE_HTTP or 500 <= code < 600


class OpenAICompatLLM:
    """最小可用的 OpenAI 兼容客户端：失败指数退避重试，空响应显式报错。"""

    def __init__(self, cfg: Config, role: str = "main", timeout: float = 120.0,
                 retries: int = 2, backoff_s: float = 0.5, max_backoff_s: float = 8.0):
        if cfg.offline:
            raise ValueError("Config 缺少 base_url/api_key/model，无法使用在线模型；请用 ScriptedLLM 或补全 .env")
        self.cfg = cfg
        self.role = role
        self.timeout = timeout
        self.retries = max(0, int(retries))
        self.backoff_s = max(0.0, float(backoff_s))
        self.max_backoff_s = max(0.0, float(max_backoff_s))
        self.model_name = cfg.model if role == "main" else (cfg.small_model or cfg.model)

    def complete(self, messages: Sequence[Message], tools: list[dict] | None = None) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [m.to_openai() for m in messages],
            "temperature": 0,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url=self.cfg.base_url.rstrip("/") + "/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.cfg.api_key}",
            },
            method="POST",
        )
        last: LLMError | None = None
        for attempt in range(self.retries + 1):
            try:
                return self._send(req)
            except LLMError as e:
                last = e
                if not e.retryable or attempt >= self.retries:
                    raise
                # 指数退避：0.5s → 1s → 2s ...，封顶 max_backoff_s
                time.sleep(min(self.max_backoff_s, self.backoff_s * (2 ** attempt)))
        raise last or LLMError("LLM 调用失败（未知原因）")

    def _send(self, req: urllib.request.Request) -> LLMResponse:
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:  # 4xx/5xx 要能看到服务端原文，便于排查
            detail = e.read().decode("utf-8", errors="replace")[:800]
            raise LLMError(f"LLM HTTPError {e.code}: {detail}", retryable=_retryable_http(e.code)) from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise LLMError(f"LLM 网络错误: {e}", retryable=True) from e
        latency = (time.perf_counter() - t0) * 1000

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise LLMError(f"LLM 返回非 JSON: {raw[:200]}", retryable=True) from e
        choices = data.get("choices") or []
        if not choices:
            # 旧实现用 (choices or [{}])[0] 兜底，会把"服务端异常/限流"变成空回答，
            # 下游只看到模型"什么都没说"，根因彻底丢失。
            raise LLMError(f"LLM 返回空 choices（可能限流或服务端异常）：{raw[:200]}", retryable=True)

        choice = choices[0]
        msg = choice.get("message") or {}
        raw_calls = msg.get("tool_calls") or []
        calls = [
            ToolCall(
                id=c.get("id") or f"call_{i}",
                name=(c.get("function") or {}).get("name", ""),
                arguments=_safe_json((c.get("function") or {}).get("arguments")),
            )
            for i, c in enumerate(raw_calls)
        ]
        usage_d = data.get("usage") or {}
        usage = Usage(
            prompt_tokens=int(usage_d.get("prompt_tokens") or 0),
            completion_tokens=int(usage_d.get("completion_tokens") or 0),
        )
        message = Message(role="assistant", content=msg.get("content") or "", tool_calls=calls)
        return LLMResponse(
            message=message,
            usage=usage,
            model=self.model_name,
            latency_ms=latency,
            cost_usd=self.cfg.cost(self.model_name, usage.prompt_tokens, usage.completion_tokens),
        )


def _safe_json(s: Any) -> dict[str, Any]:
    if isinstance(s, dict):
        return s
    if not s:
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {"_raw": v}
    except json.JSONDecodeError:
        return {"_raw": str(s)}


class ScriptedLLM:
    """离线 mock：按脚本产生响应，用来做单元测试和"没有 key 也能跑通"的演示。

    每个脚本项：
        {"tool": "calculator", "args": {"expr": "1+1"}}   -> 发起一次工具调用
        {"text": "最终答案"}                              -> 结束
    脚本用尽后重复最后一项。
    """

    def __init__(self, script: list[dict[str, Any]], model_name: str = "mock-llm", latency_s: float = 0.0,
                 price: tuple[float, float] | None = None):
        self.script = script or [{"text": "(empty script)"}]
        self.model_name = model_name
        self.calls = 0
        self.seen: list[list[Message]] = []
        # 模拟单步耗时：用于离线复现 P95 端到端延迟实验（默认 0，不影响单测）
        self.latency_s = latency_s
        # 单价（USD/1M tokens）。给了才有成本数字，否则演示里的 cost 全是 0，讲不清"省了多少钱"
        self.price = price

    def complete(self, messages: Sequence[Message], tools: list[dict] | None = None) -> LLMResponse:
        self.seen.append(list(messages))
        spec = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if self.latency_s > 0:
            time.sleep(self.latency_s)
        prompt_tokens = sum(len(m.content or "") for m in messages) // 3
        if "tool" in spec:
            call = ToolCall(id=f"call_{self.calls}", name=spec["tool"], arguments=spec.get("args") or {})
            msg = Message(role="assistant", content=spec.get("text", ""), tool_calls=[call])
        else:
            msg = Message(role="assistant", content=spec.get("text", ""))
        usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=max(1, len(msg.content) // 3))
        cost = 0.0
        if self.price:
            cost = (usage.prompt_tokens * self.price[0] + usage.completion_tokens * self.price[1]) / 1_000_000
        return LLMResponse(message=msg, usage=usage, model=self.model_name, latency_ms=0.0, cost_usd=cost)


class Router:
    """主模型 / 实用小模型的分流器。agent 只依赖这个接口。"""

    def __init__(self, cfg: Config, main: LLM | None = None, utility: LLM | None = None):
        self.cfg = cfg
        self.main = main or OpenAICompatLLM(cfg, role="main")
        self.utility = utility or OpenAICompatLLM(cfg, role="utility")
        self.utility_calls = 0
        self.main_calls = 0

    def complete(self, messages: Sequence[Message], tools: list[dict] | None = None, role: str = "main") -> LLMResponse:
        if role == "utility":
            self.utility_calls += 1
            return self.utility.complete(messages, None)
        self.main_calls += 1
        return self.main.complete(messages, tools)

    def summarize(self, text: str, max_tokens: int = 300) -> str | None:
        """用 utility 小模型做工具输出摘要；失败（离线/报错）时返回 None 由调用方降级。"""
        if not text.strip():
            return None
        prompt = [
            Message(
                role="system",
                content="你是上下文压缩器。把下面的工具输出压成要点，保留数字、路径、错误信息，不要评论。",
            ),
            Message(role="user", content=text[:20000]),
        ]
        try:
            resp = self.complete(prompt, None, role="utility")
            return (resp.message.content or "").strip() or None
        except Exception:
            return None
