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
from .config import Config
from .types import RunResult


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

        for layer in self.layers():
            for name in layer:
                node = self.nodes[name]
                if prev_status.get(name, ("", ""))[0] == "completed":
                    outputs[name] = prev_status[name][1]
                    continue

                task = node.task
                if node.deps:
                    ctx = "\n\n".join(
                        f"### {d} 的输出\n{outputs.get(d, '')[:2000]}" for d in node.deps
                    )
                    task = (node.template or "{ctx}\n\n任务：{task}").format(ctx=ctx, task=node.task)

                agent = agent_factory(node)
                res = agent.run(task, run_id=f"{dag_id}:{name}")
                outputs[name] = res.answer
                results[name] = res
                if checkpoint is not None:
                    checkpoint.mark_node(dag_id, name, "completed", res.answer)

        return {"outputs": outputs, "results": results, "dag_id": dag_id}

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
