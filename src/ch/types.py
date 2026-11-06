"""核心数据结构。刻意保持扁平，方便序列化进 checkpoint。"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


def new_id(prefix: str = "id") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ToolCall":
        return ToolCall(id=d["id"], name=d["name"], arguments=d.get("arguments") or {})


@dataclass
class Message:
    role: str  # system | user | assistant | tool
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    # pinned=True 的消息永不参与裁剪（system prompt、原始目标）
    pinned: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def to_openai(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role}
        if self.role == "assistant" and self.tool_calls:
            d["content"] = self.content or None
            d["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {
                        "name": c.name,
                        "arguments": _json_dumps(c.arguments),
                    },
                }
                for c in self.tool_calls
            ]
            return d
        d["content"] = self.content
        if self.role == "tool" and self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        return d

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "pinned": self.pinned,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Message":
        return Message(
            role=d["role"],
            content=d.get("content") or "",
            tool_calls=[ToolCall.from_dict(c) for c in d.get("tool_calls") or []],
            tool_call_id=d.get("tool_call_id"),
            name=d.get("name"),
            pinned=bool(d.get("pinned")),
            meta=d.get("meta") or {},
        )


def _json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens}


@dataclass
class LLMResponse:
    message: Message
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0


@dataclass
class ToolResult:
    call_id: str
    name: str
    ok: bool
    output: str = ""
    error: str | None = None
    # 被压缩掉的字符数 / 原文归档引用：需要能说明"压缩后仍可回溯"
    omitted_chars: int = 0
    ref: str | None = None

    def to_message(self) -> Message:
        body = self.output if self.ok else f"ERROR: {self.error}"
        if self.ref:
            body += f"\n[output compressed, {self.omitted_chars} chars archived at ref={self.ref}]"
        return Message(role="tool", content=body, tool_call_id=self.call_id, name=self.name)


@dataclass
class StepRecord:
    step: int
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    tools: list[str] = field(default_factory=list)
    note: str = ""
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "latency_ms": round(self.latency_ms, 1),
            "tools": self.tools,
            "note": self.note,
            "ts": self.ts,
        }


@dataclass
class RunResult:
    answer: str
    stop_reason: str  # final | max_steps | repeated_tool | error
    steps: int
    usage: Usage
    cost_usd: float
    trace: list[StepRecord] = field(default_factory=list)
    run_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "answer": self.answer,
            "stop_reason": self.stop_reason,
            "steps": self.steps,
            "usage": self.usage.to_dict(),
            "cost_usd": round(self.cost_usd, 6),
            "trace": [r.to_dict() for r in self.trace],
        }
