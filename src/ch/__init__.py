"""ContextHarness: a minimal, observable agent runtime.

模块地图（每个模块都对应一个具体的工程问题）：

    tokens     token 估算（无第三方依赖）
    budget     Token 预算分配器 —— 上下文窗口里该放什么
    compress   工具输出压缩器 —— tool result 爆炸
    memory     三层记忆：工作 / 情景 / 语义
    tools      工具注册表 + Schema 校验 + 白名单
    agent      主循环：目标重注入、重复调用检测、停机条件
    dag        DAG 编排 + checkpoint 断点续跑
    skill      能力封装（MCP 之上的复用层）
    mcp_client MCP stdio 客户端
    trace      每步 trace：token / 延迟 / 成本
    evaluate   任务集执行 + 指标（完成率 / 步数 / token / 成本）

v2 Super-agent（双层架构，参考 DeepSeek Super-agent 的设计思路自己实现）：

    filter         两级消息过滤：规则层零成本排噪，灰区才调小模型
    watchdog       停滞/跑偏检测：确定性判断，不调模型（它是成本控制器）
    router_worker  Router–Worker 双层：任务路由 + 独立上下文 Worker 池 + 消息直达 + 背压
    supervisor     两种模式对照：单 loop 串行 vs Filter+Router+WorkerPool

v2.1（embedding 可插拔 / 异步 Worker / 存算分离）：

    embedding      三种实现：本地哈希（默认）/ 远端 / 熔断降级，带 sqlite 缓存
    worker_store   Worker 状态库：元信息落库，执行进度在 checkpoint（不重复存）

"""

__version__ = "0.3.0"

# 顶层再导出：让 `from ch import Agent` 这种最自然的写法可用，
# 同时保持各子模块可单独导入（它们之间只用相对导入，不反向依赖包根）。
from .agent import Agent  # noqa: E402
from .budget import TokenBudget  # noqa: E402
from .config import Config  # noqa: E402
from .dag import DAG, Node  # noqa: E402
from .memory import MemoryManager  # noqa: E402
from .types import Message, ToolCall  # noqa: E402

__all__ = [
    "Agent",
    "TokenBudget",
    "Config",
    "DAG",
    "Node",
    "MemoryManager",
    "Message",
    "ToolCall",
    "__version__",
]
