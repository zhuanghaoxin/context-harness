"""工具层：注册表 + Schema 校验 + 白名单 + 危险操作审批。

常见问题："模型幻觉出一个不存在的工具/参数，你怎么处理？"
本文件的答案：
  1. 只把白名单内的工具 schema 发给模型（减少幻觉面）；
  2. 调用前用 JSON-Schema 子集校验 required/type/enum，校验失败**不抛异常给模型**，
     而是返回结构化错误，让它自己重试（这是最有效的纠错手段）；
  3. dangerous 工具必须经过审批回调，默认拒绝（评测/无人值守场景关死）；
  4. 工具执行异常被捕获并转成字符串，绝不让一个工具把整个 run 打挂。
"""
from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .types import ToolResult

ApprovalFn = Callable[[str, dict[str, Any]], bool]


def _deny_all(name: str, args: dict[str, Any]) -> bool:
    return False


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., str]
    dangerous: bool = False

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ValidationError(Exception):
    path: str
    reason: str
    tool: str = ""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.tool}.{self.path}: {self.reason}"


def validate(schema: dict[str, Any], value: Any, path: str = "", tool: str = "") -> None:
    """JSON-Schema 子集校验：type / required / enum / items / properties。"""
    t = schema.get("type")
    if t == "object":
        if not isinstance(value, dict):
            raise ValidationError(path or "$", f"expected object, got {type(value).__name__}", tool)
        for req in schema.get("required", []):
            if req not in value:
                raise ValidationError(req, "missing required argument", tool)
        props = schema.get("properties", {})
        for k, v in value.items():
            if k in props:
                validate(props[k], v, f"{path}.{k}" if path else k, tool)
            elif schema.get("additionalProperties") is False:
                raise ValidationError(k, "unexpected argument", tool)
    elif t == "string":
        if not isinstance(value, str):
            raise ValidationError(path, f"expected string, got {type(value).__name__}", tool)
    elif t == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValidationError(path, f"expected number, got {type(value).__name__}", tool)
    elif t == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValidationError(path, f"expected integer, got {type(value).__name__}", tool)
    elif t == "boolean":
        if not isinstance(value, bool):
            raise ValidationError(path, f"expected boolean, got {type(value).__name__}", tool)
    elif t == "array":
        if not isinstance(value, list):
            raise ValidationError(path, f"expected array, got {type(value).__name__}", tool)
        item_schema = schema.get("items")
        if item_schema:
            for i, item in enumerate(value):
                validate(item_schema, item, f"{path}[{i}]", tool)
    if "enum" in schema and value not in schema["enum"]:
        raise ValidationError(path, f"value {value!r} not in enum {schema['enum']}", tool)


@dataclass
class ToolRegistry:
    root: Path = field(default_factory=Path.cwd)
    approval: ApprovalFn = _deny_all
    whitelist: set[str] | None = None
    _tools: dict[str, ToolSpec] = field(default_factory=dict)
    call_count: dict[str, int] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> "ToolRegistry":
        self._tools[spec.name] = spec
        return self

    def names(self) -> list[str]:
        if self.whitelist is None:
            return sorted(self._tools)
        return sorted(self.whitelist & set(self._tools))

    def schemas(self) -> list[dict[str, Any]]:
        return [self._tools[n].to_openai_schema() for n in self.names()]

    def call(self, name: str, args: dict[str, Any], call_id: str = "") -> ToolResult:
        cid = call_id or f"call_{name}"
        if name not in self.names():
            return ToolResult(cid, name, False, error=f"工具 '{name}' 不在可用列表中，可用：{self.names()}")
        spec = self._tools[name]
        try:
            validate(spec.parameters, args, tool=name)
        except ValidationError as e:
            # 结构化错误返回给模型，由它自己修正参数（不要 try/except 掉整个 loop）
            return ToolResult(
                cid,
                name,
                False,
                error=f"参数校验失败：{e}。请修正后重试。schema={json.dumps(spec.parameters, ensure_ascii=False)[:400]}",
            )
        if spec.dangerous and not self.approval(name, args):
            return ToolResult(cid, name, False, error=f"危险工具 '{name}' 未获审批，已拒绝执行。")

        self.call_count[name] = self.call_count.get(name, 0) + 1
        try:
            out = spec.func(**args)
            return ToolResult(cid, name, True, output=str(out))
        except Exception as e:  # 工具内部异常不打断 agent
            return ToolResult(cid, name, False, error=f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------- builtin tools


def _safe_path(root: Path, p: str) -> Path:
    """把 p 解析到 root 内；用 path 组件判断，而不是字符串前缀。

    字符串前缀判断会把 /tmp/work-evil 误判为 /tmp/work 的子路径。
    """
    base = root.resolve()
    path = (base / p).resolve()
    if path != base and base not in path.parents:
        raise PermissionError(f"路径逃逸：{p} 不在工作目录 {root} 内")
    return path


def tool_read_file(root: Path) -> ToolSpec:
    def run(path: str, max_bytes: int = 20000) -> str:
        p = _safe_path(root, path)
        text = p.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_bytes:
            return text[:max_bytes] + f"\n...[文件被截断，共 {len(text)} 字节]"
        return text

    return ToolSpec(
        name="read_file",
        description="读取工作目录内的文本文件，返回内容。",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}, "max_bytes": {"type": "integer"}},
            "required": ["path"],
        },
        func=run,
    )


def tool_write_file(root: Path) -> ToolSpec:
    def run(path: str, content: str) -> str:
        p = _safe_path(root, path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"已写入 {p}（{len(content)} 字符）"

    return ToolSpec(
        name="write_file",
        description="在工作目录内创建或覆盖文件。",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
        func=run,
        dangerous=True,
    )


def tool_list_dir(root: Path) -> ToolSpec:
    def run(path: str = ".") -> str:
        p = _safe_path(root, path)
        entries = sorted(p.iterdir())
        return "\n".join(f"{'d' if e.is_dir() else 'f'} {e.name}" for e in entries) or "(空目录)"

    return ToolSpec(
        name="list_dir",
        description="列出工作目录内某个目录下的文件。",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        func=run,
    )


_SAFE_EXPR = re.compile(r"^[\d\s\+\-\*/%().,eE]+$")


def tool_calculator() -> ToolSpec:
    def run(expr: str) -> str:
        if not _SAFE_EXPR.match(expr):
            raise ValueError("只允许纯算术表达式")
        return str(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307 - 已用白名单正则收敛

    return ToolSpec(
        name="calculator",
        description="计算纯算术表达式，例如 (1+2)*3。",
        parameters={
            "type": "object",
            "properties": {"expr": {"type": "string"}},
            "required": ["expr"],
        },
        func=run,
    )


def tool_http_get() -> ToolSpec:
    def run(url: str, max_bytes: int = 5000) -> str:
        if not url.startswith(("http://", "https://")):
            raise ValueError("只支持 http/https")
        req = urllib.request.Request(url, headers={"User-Agent": "context-harness/0.1"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.read(max_bytes).decode("utf-8", errors="replace")

    return ToolSpec(
        name="http_get",
        description="发起 GET 请求并返回响应体的前若干字节。",
        parameters={
            "type": "object",
            "properties": {"url": {"type": "string"}, "max_bytes": {"type": "integer"}},
            "required": ["url"],
        },
        func=run,
        dangerous=True,
    )


def tool_shell(root: Path) -> ToolSpec:
    import subprocess

    def run(command: str, timeout: int = 30) -> str:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = (proc.stdout or "")[-8000:]
        err = (proc.stderr or "")[-2000:]
        return f"exit={proc.returncode}\nstdout:\n{out}\nstderr:\n{err}"

    return ToolSpec(
        name="shell",
        description="以工作目录为初始 cwd 执行 shell 命令；命令本身不受 cwd 限制，属高危工具。",
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}},
            "required": ["command"],
        },
        func=run,
        dangerous=True,
    )


def default_registry(
    root: str | Path = ".",
    approval: ApprovalFn | None = None,
    whitelist: set[str] | None = None,
    include_shell: bool = False,
) -> ToolRegistry:
    """默认工具集：默认不含 shell（危险），需要时显式打开 include_shell=True。"""
    r = ToolRegistry(root=Path(root), approval=approval or _deny_all, whitelist=whitelist)
    r.register(tool_calculator())
    r.register(tool_read_file(r.root))
    r.register(tool_write_file(r.root))
    r.register(tool_list_dir(r.root))
    r.register(tool_http_get())
    if include_shell:
        r.register(tool_shell(r.root))
    return r
