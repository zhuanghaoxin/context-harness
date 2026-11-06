"""v2.1 的测试：可插拔 embedding（远端/缓存/熔断） + Worker 异步与存算分离。

远端 embedding 用本地 HTTP stub 验证，不需要 key，也不需要联网。
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import shutil
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ch.agent import Agent
from ch.config import Config
from ch.embedding import (
    EmbeddingCache,
    HashingEmbedder,
    RemoteEmbedder,
    ResilientEmbedder,
    cosine,
    make_embedder,
)
from ch.llm import Router, ScriptedLLM
from ch.memory import EpisodicStore
from ch.router_worker import WorkerPool
from ch.tools import default_registry
from ch.worker_store import INTERRUPTED, WorkerRecord, WorkerStore


@contextlib.contextmanager
def tmpdir():
    d = tempfile.mkdtemp(prefix="ch-emb-")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------ 远端服务 stub


def _fake_vec(text: str, dim: int = 8) -> list[float]:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    v = [((h[i % len(h)] / 255.0) - 0.5) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 - http.server 的命名约定
        n = int(self.headers.get("Content-Length", 0) or 0)
        payload = json.loads(self.rfile.read(n) or b"{}")
        inputs = payload.get("input") or []
        self.server.calls += 1  # type: ignore[attr-defined]
        self.server.last_model = payload.get("model")  # type: ignore[attr-defined]
        body = json.dumps(
            {"data": [{"embedding": _fake_vec(t), "index": i} for i, t in enumerate(inputs)]}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # 静音
        pass


@contextlib.contextmanager
def stub_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.calls = 0
    srv.last_model = None
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=2)


def _closed_port() -> int:
    """拿一个刚关掉的端口，用来模拟远端不可用。"""
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class TestRemoteEmbedding(unittest.TestCase):
    def test_remote_embedder_returns_vectors_and_dim(self):
        with stub_server() as srv:
            with tmpdir() as d:
                e = RemoteEmbedder(f"http://127.0.0.1:{srv.server_address[1]}", model="stub-emb",
                                   cache=EmbeddingCache(f"{d}/c.sqlite3"))
                v = e.embed("上下文预算不够用")
                self.assertEqual(len(v), 8)
                self.assertEqual(e.dim, 8)
                self.assertEqual(srv.last_model, "stub-emb")

    def test_second_call_hits_cache_without_http(self):
        with stub_server() as srv:
            with tmpdir() as d:
                cache = EmbeddingCache(f"{d}/c.sqlite3")
                e = RemoteEmbedder(f"http://127.0.0.1:{srv.server_address[1]}", model="stub", cache=cache)
                e.embed("同一条消息")
                e.embed("同一条消息")
                self.assertEqual(srv.calls, 1, "第二次应当命中缓存，不再发请求")
                self.assertEqual(cache.hits, 1)

    def test_cache_persists_across_instances(self):
        with stub_server() as srv:
            with tmpdir() as d:
                p = f"{d}/c.sqlite3"
                url = f"http://127.0.0.1:{srv.server_address[1]}"
                RemoteEmbedder(url, model="stub", cache=EmbeddingCache(p)).embed("持久化")
                RemoteEmbedder(url, model="stub", cache=EmbeddingCache(p)).embed("持久化")
                self.assertEqual(srv.calls, 1)

    def test_http_error_becomes_embedding_unavailable(self):
        from ch.embedding import EmbeddingUnavailable

        e = RemoteEmbedder(f"http://127.0.0.1:{_closed_port()}", model="stub", retries=0, timeout=2.0)
        with self.assertRaises(EmbeddingUnavailable):
            e.embed("x")

    def test_resilient_degrades_to_local_and_stays(self):
        with tmpdir() as d:
            primary = RemoteEmbedder(f"http://127.0.0.1:{_closed_port()}", model="stub",
                                     retries=0, timeout=2.0, cache=EmbeddingCache(f"{d}/c.sqlite3"))
            e = ResilientEmbedder(primary, HashingEmbedder(dim=64))
            v1 = e.embed("降级前")
            self.assertTrue(e.degraded)
            self.assertEqual(e.failures, 1)
            self.assertEqual(len(v1), 64)
            v2 = e.embed("降级后")
            self.assertEqual(len(v2), 64)
            self.assertIn("degraded", e.name)
            self.assertEqual(e.failures, 1, "熔断后不应再尝试远端")

    def test_make_embedder_picks_local_when_no_model(self):
        cfg = Config(embedding_model="")
        e = make_embedder(cfg)
        self.assertIsInstance(e, HashingEmbedder)

    def test_make_embedder_picks_local_when_offline(self):
        cfg = Config(embedding_model="text-embedding-3-small")  # 无 base_url/key -> offline
        self.assertIsInstance(make_embedder(cfg), HashingEmbedder)

    def test_make_embedder_remote_when_configured(self):
        cfg = Config(base_url="https://api.example.com/v1", api_key="k", model="m",
                     embedding_model="text-embedding-3-small")
        e = make_embedder(cfg)
        self.assertIsInstance(e, ResilientEmbedder)
        self.assertIsInstance(e.primary, RemoteEmbedder)

    def test_embedding_endpoint_can_differ_from_chat(self):
        """chat 端点没有 /embeddings（如 DeepSeek）时，embedding 必须能单独指向别的厂商。"""
        with stub_server() as server, tmpdir() as d:
            cfg = Config(base_url="https://api.deepseek.com/v1", api_key="chat-key",
                         model="deepseek-chat", embedding_model="bge-m3",
                         embedding_base_url=f"http://127.0.0.1:{server.server_address[1]}",
                         embedding_api_key="emb-key",
                         embedding_cache_path=f"{d}/cache.sqlite3")
            e = make_embedder(cfg)
            self.assertEqual(len(e.embed("独立端点")), 8)
            self.assertEqual(server.calls, 1)
            self.assertFalse(getattr(e, "degraded", True))


class TestEpisodicDimGuard(unittest.TestCase):
    def test_vectors_of_other_dimension_are_skipped(self):
        """中途降级会让库里混着两种维度，维度不同的向量点积会静默算错，必须跳过。"""
        with tmpdir() as d:
            store = EpisodicStore(f"{d}/m.sqlite3", embedder=HashingEmbedder(dim=512))
            store.add("预算不够用的时候先裁历史")
            self.assertEqual(len(store.search("预算不够用", k=3)), 1)
            store.embedder = HashingEmbedder(dim=32)  # 模拟换成另一个维度的实现
            self.assertEqual(store.search("预算不够用", k=3), [])
            self.assertEqual(store.last_skipped_dim, 1)
            store.close()


class TestWorkerStore(unittest.TestCase):
    def test_save_load_roundtrip(self):
        with tmpdir() as d:
            s = WorkerStore(f"{d}/w.sqlite3")
            s.save(WorkerRecord(worker_id="w1", text="任务A", status="running"))
            rec = s.load("w1")
            assert rec is not None
            self.assertEqual(rec.text, "任务A")
            self.assertEqual(rec.status, "running")
            s.save(WorkerRecord(worker_id="w1", text="任务A", status="done", answer="好"))
            self.assertEqual(s.load("w1").status, "done")  # type: ignore[union-attr]
            self.assertEqual(s.count(), 1)
            s.close()

    def test_mark_interrupted_catches_running_rows(self):
        with tmpdir() as d:
            s = WorkerStore(f"{d}/w.sqlite3")
            s.save(WorkerRecord(worker_id="a", text="x", status="running"))
            s.save(WorkerRecord(worker_id="b", text="y", status="done"))
            self.assertEqual(s.mark_interrupted(), 1)
            self.assertEqual(s.load("a").status, INTERRUPTED)  # type: ignore[union-attr]
            self.assertEqual(s.load("b").status, "done")  # type: ignore[union-attr]
            s.close()

    def test_list_by_status(self):
        with tmpdir() as d:
            s = WorkerStore(f"{d}/w.sqlite3")
            s.save(WorkerRecord(worker_id="a", text="x", status="done"))
            s.save(WorkerRecord(worker_id="b", text="y", status="failed"))
            self.assertEqual([r.worker_id for r in s.list_workers(status="failed")], ["b"])
            self.assertEqual(len(s.list_workers()), 2)
            s.close()


def make_cfg(tmp: str, **kw) -> Config:
    cfg = Config.from_env()
    cfg.base_url = ""
    cfg.api_key = ""
    cfg.model = ""
    cfg.embedding_model = ""  # 别让本地 .env 把单测拖去真调远端
    cfg.context_budget = 2000
    cfg.max_steps = 4
    cfg.db_path = f"{tmp}/ckpt.sqlite3"
    cfg.artifact_dir = tmp
    cfg.price_per_mtok["echo-llm"] = (0.15, 0.60)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def two_step_factory(cfg: Config, latency_s: float = 0.0):
    """需要两步才收敛的 Agent：第一步调工具，第二步给答案。"""

    def factory() -> Agent:
        llm = ScriptedLLM(
            [{"tool": "calculator", "args": {"expression": "6*7"}}, {"text": "答案是 42"}],
            model_name="echo-llm",
            latency_s=latency_s,
        )
        return Agent(
            router=Router(cfg, main=llm, utility=llm),
            tools=default_registry(root=".", approval=lambda n, a: False, whitelist={"calculator"}),
            cfg=cfg,
            memory=None,
        )

    return factory


class TestAsyncWorker(unittest.TestCase):
    def test_submit_returns_id_immediately_and_poll_reads_store(self):
        with tmpdir() as d:
            cfg = make_cfg(d)
            pool = WorkerPool(cfg, two_step_factory(cfg), max_workers=2,
                              store=WorkerStore(f"{d}/w.sqlite3"))
            try:
                wid = pool.submit("算一下 6*7")
                self.assertTrue(wid.startswith("w"))
                # 存算分离的证据：主进程只持 id，状态从库里读
                rec = pool.poll(wid)
                assert rec is not None
                self.assertIn(rec["status"], ("running", "done"))
                pool.wait(wid, timeout=10)
                rec = pool.poll(wid)
                self.assertEqual(rec["status"], "done")  # type: ignore[index]
                self.assertEqual(rec["answer"], "答案是 42")  # type: ignore[index]
            finally:
                pool.close()

    def test_state_survives_dropping_in_memory_workers(self):
        with tmpdir() as d:
            path = f"{d}/w.sqlite3"
            cfg = make_cfg(d)
            pool = WorkerPool(cfg, two_step_factory(cfg), max_workers=2, store=WorkerStore(path))
            wid = pool.submit("算一下 6*7")
            pool.wait(wid, timeout=10)
            pool.workers.clear()  # 模拟"这个进程不再持有任何 Worker 对象"
            self.assertIsNone(pool.workers.get(wid))
            self.assertEqual(pool.poll(wid)["status"], "done")  # type: ignore[index]
            pool.close()

            # 换个"进程"（新连接新对象）照样读得到
            other = WorkerStore(path)
            self.assertEqual(other.load(wid).status, "done")  # type: ignore[union-attr]
            other.close()

    def test_resume_continues_from_checkpoint(self):
        """第一步被 max_steps 截断，凭 worker_id 续跑到收敛。"""
        with tmpdir() as d:
            path = f"{d}/w.sqlite3"
            short = make_cfg(d, max_steps=1)
            pool = WorkerPool(short, two_step_factory(short), max_workers=1, store=WorkerStore(path))
            wid = pool.submit("算一下 6*7")
            pool.wait(wid, timeout=10)
            rec = pool.poll(wid)
            self.assertEqual(rec["status"], "done")  # type: ignore[index]
            self.assertEqual(rec["stop_reason"], "max_steps")  # type: ignore[index]
            pool.close()

            long_cfg = make_cfg(d, max_steps=6)
            pool2 = WorkerPool(long_cfg, two_step_factory(long_cfg), max_workers=1, store=WorkerStore(path))
            try:
                pool2.resume(wid)
                pool2.wait(wid, timeout=10)
                rec = pool2.poll(wid)
                self.assertEqual(rec["status"], "done")  # type: ignore[index]
                self.assertEqual(rec["stop_reason"], "final")  # type: ignore[index]
                self.assertEqual(rec["answer"], "答案是 42")  # type: ignore[index]
            finally:
                pool2.close()

    def test_resume_unknown_id_raises(self):
        with tmpdir() as d:
            cfg = make_cfg(d)
            pool = WorkerPool(cfg, two_step_factory(cfg), store=WorkerStore(f"{d}/w.sqlite3"))
            try:
                with self.assertRaises(KeyError):
                    pool.resume("w-not-exist")
            finally:
                pool.close()

    def test_concurrent_submits_all_finish(self):
        with tmpdir() as d:
            cfg = make_cfg(d)
            # 每个 LLM 调用睡 100ms，否则任务太快、根本不会重叠，并发度测不出来
            pool = WorkerPool(cfg, two_step_factory(cfg, latency_s=0.1), max_workers=4,
                              store=WorkerStore(f"{d}/w.sqlite3"))
            try:
                ids = [pool.submit(f"算一下 {i}*7") for i in range(4)]
                for wid in ids:
                    pool.wait(wid, timeout=15)
                self.assertTrue(all(pool.poll(w)["status"] == "done" for w in ids))  # type: ignore[index]
                self.assertGreaterEqual(pool.peak_concurrency, 2)
            finally:
                pool.close()


class TestVectorRouting(unittest.TestCase):
    def test_similarity_is_cosine_of_vectors(self):
        e = HashingEmbedder(dim=256)
        a, b = e.embed("上下文预算不够用"), e.embed("上下文预算不够用")
        self.assertAlmostEqual(cosine(a, b), 1.0, places=6)

    def test_paraphrase_reuse_is_detected(self):
        """同义改写：字面 n-gram 认不出来，向量余弦可以（这里用同一实现验证通路）。"""
        with tmpdir() as d:
            cfg = make_cfg(d, worker_reuse_threshold=0.5)
            pool = WorkerPool(cfg, two_step_factory(cfg), max_workers=2,
                              store=WorkerStore(f"{d}/w.sqlite3"))
            try:
                w = pool.spawn("帮我查一下 login.py 的超时报错原因")
                self.assertIsNotNone(w.vec)
                self.assertGreater(pool.score("帮我查一下 login.py 的超时报错原因", w), 0.99)
                self.assertLess(pool.score("完全不同的话题：今晚吃什么", w), 0.2)
            finally:
                pool.close()


if __name__ == "__main__":
    unittest.main()
