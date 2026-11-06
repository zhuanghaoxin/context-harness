"""T-6 回归：MCP stdio 客户端。

之前这个模块 152 行、0% 覆盖：既能说 JSON-RPC，却没有任何路径把服务端工具接进
ToolRegistry —— 文档说"MCP 已接入"，实际是不可调用。这里用假进程覆盖真实协议路径
（握手 / tools/list / tools/call / 错误 / 超时 / 注册），不启动真实子进程。
"""
from __future__ import annotations

import json
import queue
import unittest
from unittest import mock

from ch import __version__
from ch.mcp_client import MCPClient, MCPError
from ch.tools import ToolRegistry


class _Server:
    """按 JSON-RPC 规则应答的极小服务端。"""

    def __init__(self) -> None:
        self.out: queue.Queue = queue.Queue()
        self.requests: list[dict] = []
        self.notifications: list[dict] = []

    def handle(self, msg: dict) -> None:
        self.requests.append(msg)
        if "id" not in msg:
            self.notifications.append(msg)
            return
        mid = msg["id"]
        method = msg.get("method")
        if method == "initialize":
            payload = {"result": {"protocolVersion": "2024-11-05",
                                  "serverInfo": {"name": "fake", "version": "1.0.0"}}}
        elif method == "tools/list":
            payload = {"result": {"tools": [{
                "name": "echo",
                "description": "回显文本",
                "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}},
                                "required": ["text"]},
            }]}}
        elif method == "tools/call":
            if msg["params"]["name"] == "boom":
                self.out.put(json.dumps({"jsonrpc": "2.0", "id": mid,
                                         "error": {"code": -32000, "message": "nope"}}))
                return
            text = msg["params"]["arguments"].get("text", "")
            payload = {"result": {"content": [
                {"type": "text", "text": f"echo:{text}"},
                {"type": "image", "data": "ignored"},
                "raw-line",
            ]}}
        else:
            payload = {"error": {"code": -32601, "message": f"unknown {method}"}}
        payload.update({"jsonrpc": "2.0", "id": mid})
        self.out.put(json.dumps(payload))


class _Stdin:
    def __init__(self, server: _Server) -> None:
        self.server = server
        self.buf = ""

    def write(self, text: str) -> int:
        self.buf += text
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                self.server.handle(json.loads(line))
        return len(text)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class _Stdout:
    def __init__(self, q: queue.Queue) -> None:
        self.q = q

    def __iter__(self):
        return self

    def __next__(self) -> str:
        item = self.q.get()
        if item is None:
            raise StopIteration
        return item + "\n"


class _Proc:
    def __init__(self, server: _Server) -> None:
        self.server = server
        self.stdin = _Stdin(server)
        self.stdout = _Stdout(server.out)
        self._done = False

    def poll(self):
        return 0 if self._done else None

    def terminate(self) -> None:
        self._done = True
        self.server.out.put(None)

    def wait(self, timeout=None) -> int:
        return 0


class MCPClientTest(unittest.TestCase):
    def _patched(self):
        server = _Server()
        proc = _Proc(server)
        return server, proc, mock.patch("ch.mcp_client.subprocess.Popen", return_value=proc)

    def test_handshake_list_and_install(self):
        server, _proc, patcher = self._patched()
        with patcher:
            with MCPClient(["fake-server"]) as client:
                self.assertEqual(client.server_info["name"], "fake")
                schemas = client.to_openai_schemas()
                self.assertEqual(schemas[0]["function"]["name"], "echo")
                self.assertEqual(schemas[0]["function"]["parameters"]["required"], ["text"])

                reg = ToolRegistry(approval=lambda n, a: True)
                self.assertEqual(client.install_into(reg, dangerous=False), 1)
                out = reg.call("echo", {"text": "hi"})
                self.assertTrue(out.ok)
                self.assertEqual(out.output, "echo:hi\nraw-line")

        init = [r for r in server.requests if r.get("method") == "initialize"][0]
        # 握手要报真实包版本，写死 0.1.0 会让服务端看到过期客户端
        self.assertEqual(init["params"]["clientInfo"]["version"], __version__)
        self.assertTrue(any(n.get("method") == "notifications/initialized" for n in server.notifications))
        ids = [r["id"] for r in server.requests if "id" in r]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(ids), len(set(ids)))

    def test_error_response_raises(self):
        _server, _proc, patcher = self._patched()
        with patcher:
            with MCPClient(["fake-server"]) as client:
                with self.assertRaises(MCPError) as ctx:
                    client.call_tool("boom", {})
                self.assertIn("tools/call 失败", str(ctx.exception))

    def test_timeout(self):
        silent = _Server()
        silent.handle = lambda msg: None  # 永不回应
        proc = _Proc(silent)
        with mock.patch("ch.mcp_client.subprocess.Popen", return_value=proc):
            client = MCPClient(["silent"], timeout=0.15)
            with self.assertRaises(MCPError) as ctx:
                client.start()
            self.assertIn("超时", str(ctx.exception))
            client.close()

    def test_send_before_start(self):
        client = MCPClient(["never-started"])
        with self.assertRaises(MCPError):
            client.notify("x", {})


if __name__ == "__main__":
    unittest.main()
