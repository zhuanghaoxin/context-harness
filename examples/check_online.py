"""最小连通性验证：真实打一次 chat + 一次 embeddings，确认 key/端点可用。"""
from __future__ import annotations

import sys

from ch.config import Config
from ch.embedding import EmbeddingUnavailable, RemoteEmbedder
from ch.llm import OpenAICompatLLM, Router
from ch.types import Message


def main() -> int:
    cfg = Config.from_env()
    print(f"base_url={cfg.base_url}  model={cfg.model}  offline={cfg.offline}")
    if cfg.offline:
        print("FAIL: .env 没读到 key")
        return 2

    router = Router(cfg, main=OpenAICompatLLM(cfg, role="main"))
    try:
        resp = router.complete([Message(role="user", content="只回复两个字：收到")])
        print(f"chat ok: {resp.message.content.strip()[:20]!r} "
              f"tokens={resp.usage.prompt_tokens}/{resp.usage.completion_tokens} "
              f"cost=${resp.cost_usd:.6f} latency={resp.latency_ms:.0f}ms")
    except Exception as exc:  # noqa: BLE001 - 就是要看原始错误
        print(f"FAIL chat: {type(exc).__name__}: {exc}")
        return 3

    if cfg.embedding_model:
        base = cfg.embedding_base_url or cfg.base_url
        key = cfg.embedding_api_key or cfg.api_key
        emb = RemoteEmbedder(base, key, cfg.embedding_model,
                             dim=cfg.embedding_dim, timeout=cfg.embedding_timeout)
        try:
            vec = emb.embed("测试一下 embedding 端点")
            print(f"embedding ok: url={base} model={cfg.embedding_model} dim={len(vec)}")
        except EmbeddingUnavailable as exc:
            print(f"embedding 不可用（会熔断降级到哈希）: {exc}")
    else:
        print("embedding: 未配置，用本地哈希")
    return 0


if __name__ == "__main__":
    sys.exit(main())
