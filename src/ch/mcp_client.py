"""MCP stdio 客户端（最小可用实现，JSON-RPC 2.0 / newline-delimited）。

只实现 harness 真正需要的三件事：
    initialize  ->  tools/list  ->  tools/call

接入流程（"MCP 怎么接"）：
  1. transport 是 stdio，消息是**按行分隔**的 JSON-RPC 2.0；
  2. 握手：initialize（带 protocolVersion + clientInfo），随后发 notifications/initialized；
  3. tools/list 拿到 schema，转成 OpenAI function schema 后并入本地工具表；
  4. tools/call 返回 content 数组，取 text 类型拼成字符串；
  5. 读循环要按 id 匹配响应，通知（无 id）直接丢弃。
注意：Windows 下用 pipe 启动子进程可能被沙箱策略拦截，属预期现象，
本地开发环境（非沙箱）可正常运行。测试用例不覆盖真实进程启动。
"""
from __future__ import annotations

import json
import subprocess
import threading
from typing import Any

PROTOCOL_VERSION = "2024-11-05"


class MCPError(RuntimeError):
    pass


class MCPClient:
    def __init__(self, command: list[str], env: dict[str, str] | None = None, timeout: float = 20.0):
        self.command = command
        self.env = env
        self.timeout = timeout
        self.proc: subprocess.Popen | None = None
        self._next_id = 0
        self._pending: dict[int, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._reader: threading.Thread | None = None
        self.server_info: dict[str, Any] = {}

    # ------------------------------------------------------------- lifecycle
    def start(self) -> "MCPClient":
        self.proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            env=self.env,
            bufsize=1,
        )
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        resp = self.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "context-harness", "version": "0.1.0"},
            },
        )
        self.server_info = resp.get("result", {}).get("serverInfo", {})
        self.notify("notifications/initialized", {})
        return self

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin and self.proc.stdin.close()
                self.proc.terminate()
            except Exception:
                pass

    def __enter__(self) -> "MCPClient":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------- transport
    def _send(self, payload: dict[str, Any]) -> None:
        if not self.proc or not self.proc.stdin:
            raise MCPError("MCP 进程未启动")
        self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def _read_loop(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            mid = msg.get("id")
            if isinstance(mid, int):
                with self._lock:
                    self._pending[mid] = msg

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._next_id += 1
        mid = self._next_id
        self._send({"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}})
        waited = 0.0
        step = 0.02
        while waited < self.timeout:
            with self._lock:
                if mid in self._pending:
                    msg = self._pending.pop(mid)
                    if "error" in msg:
                        raise MCPError(f"{method} 失败：{msg['error']}")
                    return msg
            threading.Event().wait(step)
            waited += step
        raise MCPError(f"{method} 超时（{self.timeout}s）")

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    # ------------------------------------------------------------- tools
    def list_tools(self) -> list[dict[str, Any]]:
        resp = self.request("tools/list", {})
        return resp.get("result", {}).get("tools", [])

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        resp = self.request("tools/call", {"name": name, "arguments": arguments})
        content = resp.get("result", {}).get("content") or []
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(item.get("text", ""))
            elif isinstance(item, str):
                chunks.append(item)
        return "\n".join(chunks)

    def to_openai_schemas(self) -> list[dict[str, Any]]:
        out = []
        for t in self.list_tools():
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.get("name", ""),
                        "description": t.get("description", ""),
                        "parameters": t.get("inputSchema") or {"type": "object", "properties": {}},
                    },
                }
            )
        return out
