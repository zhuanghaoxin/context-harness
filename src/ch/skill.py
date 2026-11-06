"""Skill 注册表。

要回答的问题："MCP 和 Skill 有什么区别？"
一句话版本：
  - MCP 是**跨进程协议**，解决的是"能力怎么被外部提供"（transport + 生命周期）；
  - Skill 是**提示词层面的能力封装**，解决的是"这类任务该怎么一步步做"。
    它往往组合多个 tool、带上流程约束和验收标准。

所以 Skill 的四要素是：功能描述 / 输入 / 输出 / 依赖工具。
没有"依赖工具"这一项，Skill 就退化成一段 prompt 文本。

本模块把 Skill 注册成可注入 system prompt 的块，并在注入前校验依赖工具是否存在——
避免模型调用一个"依赖工具没装"的 Skill 然后一路幻觉下去。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class Skill:
    name: str
    description: str
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    tools: tuple[str, ...] = ()
    steps: tuple[str, ...] = ()
    acceptance: tuple[str, ...] = ()

    def to_prompt_block(self) -> str:
        parts = [f"### Skill: {self.name}", self.description]
        if self.inputs:
            parts.append("输入：" + "; ".join(f"{k}({v})" for k, v in self.inputs.items()))
        if self.outputs:
            parts.append("输出：" + "; ".join(f"{k}({v})" for k, v in self.outputs.items()))
        if self.tools:
            parts.append("依赖工具：" + ", ".join(self.tools))
        if self.steps:
            parts.append("步骤：" + " -> ".join(f"{i+1}.{s}" for i, s in enumerate(self.steps)))
        if self.acceptance:
            parts.append("验收：" + "; ".join(self.acceptance))
        return "\n".join(parts)


@dataclass
class SkillRegistry:
    skills: dict[str, Skill] = field(default_factory=dict)

    def register(self, skill: Skill) -> "SkillRegistry":
        self.skills[skill.name] = skill
        return self

    def from_mcp_tool(self, tool: dict[str, Any]) -> Skill:
        """把一个 MCP tool 包装成单步 Skill，便于统一注入。"""
        fn = tool.get("function") or tool
        name = fn.get("name", "unnamed")
        # 兼容两种 schema 命名：MCP 原生用 inputSchema，OpenAI function 用 parameters
        schema = fn.get("parameters") or fn.get("inputSchema") or {}
        return Skill(
            name=name,
            description=fn.get("description", ""),
            inputs={k: str(v) for k, v in (schema.get("properties") or {}).items()},
            outputs={"result": "string"},
            tools=(name,),
            steps=(f"调用 {name}",),
        )

    def validate(self, available_tools: Iterable[str]) -> list[str]:
        """返回依赖缺失的 Skill 名。这些 Skill 不应被注入上下文。"""
        avail = set(available_tools)
        return [s.name for s in self.skills.values() if not set(s.tools) <= avail]

    def prompt_block(self, available_tools: Iterable[str]) -> str:
        broken = set(self.validate(available_tools))
        usable = [s for s in self.skills.values() if s.name not in broken]
        if not usable:
            return ""
        body = "\n\n".join(s.to_prompt_block() for s in usable)
        return "# 可用 Skill\n" + body


def default_skills() -> SkillRegistry:
    return (
        SkillRegistry()
        .register(
            Skill(
                name="code_review",
                description="审查一段代码的正确性、边界与可维护性。",
                inputs={"path": "string 文件路径", "focus": "string 关注点，可为空"},
                outputs={"findings": "按严重度排序的问题列表"},
                tools=("read_file", "list_dir"),
                steps=("读文件", "定位问题", "按严重度排序输出"),
                acceptance=("每个问题给出文件与行号", "区分 blocker 与建议"),
            )
        )
        .register(
            Skill(
                name="data_check",
                description="核对一组数字的口径与一致性，输出异常项。",
                inputs={"exprs": "string 待核算的表达式或数字"},
                outputs={"anomalies": "异常列表"},
                tools=("calculator",),
                steps=("重算关键数字", "对比给出差异", "标注口径不一致"),
            )
        )
    )
