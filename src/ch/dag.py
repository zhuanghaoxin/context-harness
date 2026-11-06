"""DAG 编排：把长任务拆成有依赖的节点，节点级 checkpoint。

要回答的问题："Multi-Agent / 工作流怎么编排？失败了怎么续跑？"
本文件的取舍说明（可直接用于设计评审）：
  - 先做**确定性 DAG**，不做模型自主规划：预算可控、可复现、好评测。
    （让模型自由规划拓扑在生产上极难调，先固定骨架是工程上的正确选择。）
  - 拓扑分层执行：同层节点无依赖，可以并行；这里先用串行实现保证
    结果可复现，接口留成 `run_parallel` 的形状。
  - 每个节点完成即 checkpoint，重跑时跳过 completed 节点。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .agent import Agent
from .budget import truncate_to_tokens
from .config import Config
from .types import RunResult

# 上游节点输出注入下游时的 token 预算（此前是 [:2000] 字符硬切，中文会超预算且切口随机）
UPSTREAM_TOKEN_BUDGET = 800


class DAGRunError(RuntimeError):
    """有节点失败：失败节点已落库为 failed，下游落库为 skipped。"""

    def __init__(self, message: str, failures: dict[str, Exception], skipped: list[str],
                 outputs: dict[str, str], results: dict[str, RunResult]):
        super().__init__(message)
        self.failures = failures
        self.skipped = skipped
        self.outputs = outputs
        self.results = results


@dataclass
class Node:
    name: str
    task: str
    deps: tuple[str, ...] = ()
    # 上游输出注入模板，例如 "上下文：{research}\n现在做：写摘要"
    template: str | None = None


@dataclass
class DAG:
    name: str = "dag"
    nodes: dict[str, Node] = field(default_factory=dict)

    def add(self, node: Node) -> "DAG":
        if node.name in self.nodes:
            raise ValueError(f"节点重名：{node.name}")
        self.nodes[node.name] = node
        return self

    # ------------------------------------------------------------- 校验
    def validate(self) -> None:
        for n in self.nodes.values():
            for d in n.deps:
                if d not in self.nodes:
                    raise ValueError(f"节点 {n.name} 依赖不存在的节点 {d}")
        # 环检测（DFS 三色标记）
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {k: WHITE for k in self.nodes}

        def visit(name: str) -> None:
            color[name] = GRAY
            for d in self.nodes[name].deps:
                if color[d] == GRAY:
                    raise ValueError(f"检测到环：{name} -> {d}")
                if color[d] == WHITE:
                    visit(d)
            color[name] = BLACK

        for k in self.nodes:
            if color[k] == WHITE:
                visit(k)

    def layers(self) -> list[list[str]]:
        """拓扑分层：同层可并行。"""
        self.validate()
        done: set[str] = set()
        out: list[list[str]] = []
        remaining = set(self.nodes)
        while remaining:
            layer = sorted(n for n in remaining if all(d in done for d in self.nodes[n].deps))
            if not layer:
                raise ValueError("拓扑排序失败（存在环）")
            out.append(layer)
            done |= set(layer)
            remaining -= set(layer)
        return out

    # ------------------------------------------------------------- 执行
    def run(
        self,
        agent_factory: Callable[[Node], Agent],
        dag_id: str | None = None,
        checkpoint: Any = None,
        resume: bool = False,
    ) -> dict[str, Any]:
        dag_id = dag_id or self.name
        outputs: dict[str, str] = {}
        results: dict[str, RunResult] = {}
        prev_status = checkpoint.node_status(dag_id) if (checkpoint and resume) else {}

        completed: set[str] = set()
        failed: dict[str, Exception] = {}
        skipped: list[str] = []
        for layer in self.layers():
            for name in layer:
                node = self.nodes[name]
                if prev_status.get(name, ("", ""))[0] == "completed":
                    outputs[name] = prev_status[name][1]
                    completed.add(name)
                    continue

                # 依赖没全部完成（上游失败或被跳过）→ 本节点不执行，落 skipped 而不是静默跳过
                if any(d not in completed for d in node.deps):
                    skipped.append(name)
                    if checkpoint is not None:
                        checkpoint.mark_node(dag_id, name, "skipped", "上游节点未完成")
                    continue

                task = node.task
                if node.deps:
                    parts: list[str] = []
                    for d in node.deps:
                        shown = truncate_to_tokens(
                            outputs.get(d, ""),
                            UPSTREAM_TOKEN_BUDGET,
                            marker="\n...[上游输出已按预算截断]",
                        )
                        parts.append(f"### {d} 的输出\n{shown}")
                    ctx = "\n\n".join(parts)
                    task = (node.template or "{ctx}\n\n任务：{task}").format(ctx=ctx, task=node.task)

                try:
                    agent = agent_factory(node)
                    # resume 透传给 Agent：节点内部半途中断时，重跑从断点继续而不是重放工具
                    res = agent.run(task, run_id=f"{dag_id}:{name}", resume=resume)
                except Exception as e:  # noqa: BLE001 - 节点失败要落库并让下游跳过，而不是裸抛
                    failed[name] = e
                    if checkpoint is not None:
                        checkpoint.mark_node(dag_id, name, "failed", f"{type(e).__name__}: {e}"[:500])
                    continue

                outputs[name] = res.answer
                results[name] = res
                completed.add(name)
                if checkpoint is not None:
                    checkpoint.mark_node(dag_id, name, "completed", res.answer)

        if failed:
            raise DAGRunError(
                f"DAG {dag_id} 有 {len(failed)} 个节点失败：{sorted(failed)}；已跳过下游 {skipped}",
                failures=failed,
                skipped=skipped,
                outputs=outputs,
                results=results,
            )
        return {"outputs": outputs, "results": results, "dag_id": dag_id, "skipped": skipped}

    def mermaid(self) -> str:
        lines = ["graph TD"]
        for n in self.nodes.values():
            lines.append(f"  {n.name}[\"{n.name}\"]")
            for d in n.deps:
                lines.append(f"  {d} --> {n.name}")
        return "\n".join(lines)


def demo_dag() -> DAG:
    """示例：调研 -> 起草 -> 校对。写进 README 当架构图。"""
    return (
        DAG(name="research-write")
        .add(Node("research", "收集 3 条关于 agent 上下文管理的关键事实，逐条列出。"))
        .add(Node("draft", "基于调研写 150 字摘要。", deps=("research",)))
        .add(Node("review", "检查摘要是否有事实错误，输出修正后的最终版。", deps=("draft",)))
    )
