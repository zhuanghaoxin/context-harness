# ContextHarness

一个**零依赖**的 mini agent harness：把大模型从「只会聊天」变成「能规划、能调工具、能记住、能自评」的执行器，并且**把上下文预算当成一等资源**来管理。

```bash
python examples/run_mock.py                       # 离线跑通全部机制（不装包、不联网、不需要 API key）

# 让 python -m ch.* 找得到源码（Windows PowerShell）
$env:PYTHONPATH='src'        #  bash/zsh: export PYTHONPATH=src ；也可以 pip install -e .

python -m unittest discover -s tests -t .         # 93 个单测
python -m ch.evaluate --variants baseline budget full   # 消融对照（离线回放）
```

> 定位：不是又一个 LangChain 封装，而是把 harness 里**最容易在真实生产中被卡住的 5 件事**（上下文预算、记忆、工具输出膨胀、状态持久化、评估闭环）用最少的代码写成可读、可测、可复现的工程实现。
>
> **真实模型实测数字在 [BENCHMARKS.md](BENCHMARKS.md)**（deepseek-chat，3 次重复：token/任务 -23.3%、成本 -22.9%、双层 makespan -69.8%），README 正文里的离线数字只用于验证机制。

---

## 1. 为什么要自己写

通用框架（LangChain / Dify）把决策藏进了抽象层：一旦线上出问题，很难回答「上下文满了先砍谁」「工具调用失败了怎么办」「记忆是怎么写进去的」。生产环境对 harness 的要求是：

- 说得清主循环每一步做了什么：记忆机制怎么落盘、工具失败怎么恢复、上下文超预算按什么优先级裁剪。
- 说得出**框架的取舍**而不只是「会用」：为什么不用 LangChain / Dify，代价是什么。
- 只讲「支持了 N 个工具」没有说服力；要能给出**失败率、成本、延迟、坏 case 归因**。

所以本项目的每个模块都对应一个具体的工程问题，并且都能跑出一条可验证的数字（见第 4 节）。

## 2. 架构

```
                    ┌──────────────────────── Agent.run() 主循环 ────────────────────────┐
                    │                                                                    │
   task ──────────► │  1. 目标重注入（每 N 步把目标贴回上下文尾部，抗长上下文注意力衰减）    │
                    │  2. TokenBudget.assemble()  ──► 四类配额：system/task | 记忆 | 历史   │
                    │        │                         | 工具输出，超限按优先级裁剪          │
                    │        ▼                                                           │
                    │  3. Router.complete()  ──► 主模型（决策）/ 小模型（utility 摘要）    │
                    │        │                                                           │
                    │        ├─ 无 tool_calls ──► final，结束                            │
                    │        └─ 有 tool_calls                                            │
                    │              ├─ ToolRegistry.call()  ──► 参数校验/白名单/审批/异常    │
                    │              │                            全部返回结构化 ToolResult   │
                    │              ├─ 重复调用检测（同参数重试 → 警告 → 停机）              │
                    │              ├─ ToolOutputCompressor ──► 头尾保留 + 全文归档可回查     │
                    │              └─ CheckpointStore.save() ──► 每步落盘，可续跑/可回滚     │
                    │                                                                    │
                    │  4. 每步 Tracer 记录 token/成本/延迟/工具名 ──► trace + 汇总表       │
                    └────────────────────────────────────────────────────────────────────┘

   MemoryManager（三层）  ◄── recall_block(query) 注入上下文    ──►  episode 回写
   DAG（确定性编排）      ◄── 节点 = 一次 Agent.run()，上游输出注入下游，completed 跳过
   SkillRegistry / MCP    ◄── 依赖缺失的 Skill 不注入上下文（避免模型调用不存在的工具）
   evaluate               ◄── 任务集 + checker + 多档消融 → 通过率/token/成本对照表
```

## 3. 模块 ↔ 设计要点

| 模块 | 文件 | 解决什么 | 评审常关注的点 |
|---|---|---|---|
| Token 预算分配器 | `src/ch/budget.py` | 四类配额 + 按优先级裁剪（先截记忆 → 整组丢最旧 → 硬截断大 tool 输出） | 上下文窗口怎么分配？Lost in the Middle 怎么缓解？为什么按「组」丢而不是按消息丢？ |
| 工具输出压缩 | `src/ch/compress.py` | 头 60% + 标记 + 尾，全文归档到 `.artifacts/`，消息里留 ref 可回查 | tool result 爆炸怎么办？压缩丢了关键信息怎么恢复？ |
| 三层记忆 | `src/ch/memory.py` | 工作记忆=上下文；情景记忆=向量检索；语义记忆=去重/冲突更新/时间衰减 | Mem0/Zep 的差别？事实冲突谁赢？遗忘曲线怎么实现？ |
| 状态持久化 | `src/ch/checkpoint.py` | 每步落盘，`resume=True` 续跑，DAG 节点级 completed 跳过 | 长任务断点续跑、幂等、回滚怎么做？ |
| DAG 编排 | `src/ch/dag.py` | 三色 DFS 环检测 + 拓扑分层 + 上游输出注入 | 为什么先做确定性 DAG 不做自主规划？（预算可控/可复现/好评测） |
| Skill / MCP | `src/ch/skill.py`、`src/ch/mcp_client.py` | Skill 四要素（功能/输入 Schema/输出 Schema/依赖工具）；最小 MCP stdio 客户端（JSON-RPC 2.0 握手） | MCP 和 Skill 的区别？协议握手流程？依赖缺失怎么处理？ |
| 可观测 + 评估 | `src/ch/trace.py`、`src/ch/evaluate.py` | 每步 token/成本/延迟/工具；任务集 checker；多档消融 | 怎么评估一个 Agent？成本和通过率怎么同时看？ |

## 3.5 Super-agent v2：Router–Worker 双层

v1 是「一条消息 → 一个 loop 跑完」。真实工作流里两个假设都不成立：任务之间大多无关却挤在同一条历史里，前一个没跑完后一个只能排队。

v2 按 Super-agent 的三层职责拆开（设计思路参考 DeepSeek 的 Super-agent 技术文章，代码是自己写的）：

```
  消息流 ──► MessageFilter ──► TaskRouter ──┬─► Worker A（独立上下文，可并行）
             （要不要处理）    （交给谁）     ├─► Worker B
                                          └─► Worker C
                 ▲                             │
                 └──── Watchdog（每步判断：还要不要继续）◄┘
```

| 组件 | 文件 | 关键设计 |
|---|---|---|
| 两级消息过滤 | `src/ch/filter.py` | L1 规则层（正则/长度/任务动词/报错词）零成本判掉明显噪音与明显任务；**只有灰区**才调 utility 模型四分类。灰区拿不到模型判断时 **fail-open**（漏任务代价 >> 多处理一条噪音） |
| Watchdog | `src/ch/watchdog.py` | 三种卡法分开处理：`loop`（工具名+参数+输出摘要全同）→ 停机；`no_progress`（上下文不增长）→ 停机；`off_topic`（与目标重合度过低）→ 先重注入目标，连续 2 次才停。**只做确定性判断，不调模型**——它是成本控制器，且出问题时最可疑的就是模型 |
| Router–Worker | `src/ch/router_worker.py` | Router 决定「消息直达已有 Worker / 新开 Worker / 拒收」。消息直达：同话题后续消息拼进 Worker 自己的任务，不重建上下文。背压分两级：`max_workers` 管同时跑几个，`queue_factor` 管最多堆几条，超过才拒收 |
| 两模式对照 | `src/ch/supervisor.py` | `run_single`（串行，不过滤不路由）vs `run_super`（过滤+路由+并行），同一批输入、同一套 checker，直接出对照表 |

实测（`python examples/run_super.py --online`，真实 deepseek-chat，3 次重复取均值，6 条真实任务 + 4 条噪音）：

| 指标 | single | super | 变化 |
|---|---|---|---|
| 实跑 loop | 10 | 6 | 噪音 4 条零成本丢弃 |
| makespan | 21967 ms | 6639 ms | **-69.8%** |
| P95 端到端延迟 | 21403 ms | 6461 ms | **-69.8%** |
| 峰值并发 | 1 | 4 | — |
| 主动委派 | 0 | 6 | 用户没要求拆分，Router 主动分了工 |
| 总 token / 成本 | 5711 / $0.00303 | 3944 / $0.00238 | -30.9% / -21.4% |

离线回放（不加 `--online`，模型延迟模拟 120ms）是 makespan -80%：真实模型延迟方差大，并行收益被长尾吃掉一部分，所以线上数字略低。

**这些数字的边界（必须主动说清）**：任务集只有 10 条，单任务 token 波动 2650–3956，结论是**机制级**不是能力级；token 收益主要来自「不跑噪音」，不是模型更聪明；Super-agent 文章里那个 P95 15.7min vs 29.5min（-47%）是他们 300 条消息的真实模拟口径，口径不同不能直接比。完整数字与三次单跑明细见 [BENCHMARKS.md](BENCHMARKS.md)。

另一个更值得讲的发现：中性提示词下连跑五轮，**一次都没有**主动调用子 Agent；把「鼓励委派」写进提示词后五轮里有四轮用了。所以「主动找活」必须是工程机制，不能靠提示词——这就是 `filter.py` 存在的理由。

## 3.6 Super-agent v2.1：真实 embedding + 异步 Worker + 存算分离

这三件事对应原文里「多 Agent 协作必须靠工程机制」的另一半：**路由得判得准、Worker 得能异步、状态得能离开进程**。

**① Router 的话题相似度可插拔**（`src/ch/embedding.py`）

| 实现 | 何时用 | 代价 |
|---|---|---|
| `HashingEmbedder` | 默认，离线、零依赖 | 只有字面相似度，同义改写判不出 |
| `RemoteEmbedder` | 配了 `CH_EMBEDDING_MODEL` | 一次 HTTP 批量请求 + sqlite 缓存 |
| `ResilientEmbedder` | 自动包在上面一层 | 远端失败**整场熔断**，降级到哈希 |

为什么是**整场熔断**而不是逐次重试：embedding 服务挂了的时候，逐次尝试等于每条消息都等一次 30 秒超时；宁可整场降级到「判得糙但能跑完」。降级后必须做**维度守卫**：远端 1536 维、本地 512 维，长度不等时点积会静默算错，`EpisodicStore.search` 直接跳过并计数（`last_skipped_dim`）。

embedding 端点可以和 chat 端点分开（`CH_EMBEDDING_BASE_URL` / `CH_EMBEDDING_API_KEY`）：DeepSeek 这类只做 chat 的厂商没有 `/embeddings`，实测会 404 然后走熔断——所以本仓库的真实数字都是在**降级态**下跑出来的（评测照常跑完，只是话题判定用哈希）。

**② Worker 异步**：`pool.submit()` 立刻返回 `worker_id`，执行在 `ThreadPoolExecutor`；主线程只持 id，用 `poll(id)` / `wait(id)` 查状态。提交与处理解耦，路由层不必等 Worker 跑完。

**③ 存算分离**：状态拆两层——`WorkerStore`（`workers` 表）只存元信息（状态/答案/token/成本），执行进度留在 `checkpoint.runs`，`run_id` 复用 `worker_id`。演示（`examples/run_async.py`）：换一个新开的 `WorkerStore` 连接、只凭 id 就能读回完整状态；进程重启后 `mark_interrupted()` 把残留 `running` 标成 `interrupted`，再 `resume(worker_id)` 从断点接着跑。

> 注意：`WorkerStore` 存的是**元信息**，执行进度在 `checkpoint.runs`。两边都存进度会打架，这是拆分时先定死的边界。

## 4. 评测：把「我觉得更好」变成数字

三档消融（`src/ch/evaluate.py`）：

| variant | 上下文预算 | 记忆 | 用途 |
|---|---|---|---|
| `baseline` | 200000 | 无 | 朴素实现：全塞、不记、不裁 |
| `budget` | 2500 | 无 | 只开预算裁剪 |
| `full` | 2500 | 三层记忆 | 完整形态 |

**真实模型（deepseek-chat，3 次重复取均值，明细见 [BENCHMARKS.md](BENCHMARKS.md)）**

| variant | 通过率 | token/任务 | 成本/任务 | 平均步数 |
|---|---|---|---|---|
| `baseline` | 100% | 3237（2650–3956） | $0.00102 | 2.30 |
| `budget` | 100% | 3537 | $0.00112 | 2.73 |
| `full` | 100% | **2482（-23.3%）** | **$0.00079（-22.9%）** | 2.23 |

**反例（比结论更值得讲）**：只开裁剪的 `budget` 比 `baseline` 更贵（token +9.3%、成本 +8.9%、步数 2.30→2.73）。原因是上下文一裁，模型看不到先前的工具结果，只能重新调一次工具把信息取回——省下的 prompt token 被多出的 step 吃掉。所以「省 token」不是目标，**单位任务成本**才是，裁剪必须和记忆一起开。

```bash
# 离线回放：验证管线正确性（指标口径、停机原因、checker、裁剪是否生效）
python -m ch.evaluate --tasks tasks/tasks.json --variants baseline budget full

# v2：单 loop 串行 vs Filter+Router+WorkerPool 双层对照
python examples/run_super.py

# 接真实模型：产生可对外披露的指标（需要 .env）
python examples/check_online.py            # 先自检 key/端点，一次 chat + 一次 embeddings
python -m ch.evaluate --variants baseline budget full --online --out .artifacts/real-eval.json
python examples/run_super.py --online
```

指标：`pass_rate / tokens_per_task / cost_per_task / avg_steps / stop_reasons`，明细写到 `.artifacts/eval-*.json`（含每条任务的回答，便于坏 case 归因）。

> ⚠️ 离线回放用 `ScriptedLLM` 重放 `tasks.json` 里的 `script`，所以通过率一定是满的——它验证的是**管线**，不是模型效果。对外引用的数字必须来自 `--online`。

## 5. 目录

```
context-harness/
├── src/ch/
│   ├── types.py       消息/ToolCall/ToolResult/RunResult 等数据结构
│   ├── tokens.py      中英混排估 token（CJK 1 字/token，其余 4 字符/token）
│   ├── config.py      环境变量/成本表/预算份额
│   ├── budget.py      上下文预算分配与裁剪
│   ├── compress.py    工具输出压缩 + 归档回查
│   ├── memory.py      三层记忆（embedding 可插拔 + sqlite）
│   ├── embedding.py   三种 embedding：哈希 / 远端 / 熔断降级，带 sqlite 缓存
│   ├── tools.py       工具注册表 + 参数校验 + 审批 + 内置工具
│   ├── llm.py         OpenAI 兼容客户端 / 离线 ScriptedLLM / 主+小模型路由
│   ├── agent.py       主循环、目标重注入、重复调用保护、停机原因
│   ├── checkpoint.py  sqlite 断点续跑
│   ├── dag.py         确定性 DAG 编排
│   ├── skill.py       Skill 注册表（四要素）
│   ├── mcp_client.py  最小 MCP stdio 客户端
│   ├── trace.py       每步 trace + 汇总表
│   ├── evaluate.py    任务集评测与消融
│   ├── filter.py      两级消息过滤（规则层 + 灰区模型）
│   ├── watchdog.py    停滞/跑偏检测（loop / no_progress / off_topic）
│   ├── router_worker.py  Router–Worker 双层：路由 + Worker 池 + 消息直达 + 背压
│   ├── worker_store.py  Worker 状态库（存算分离：元信息落库，进度在 checkpoint）
│   └── supervisor.py  两模式对照：single vs super
├── tasks/tasks.json   10 条自建任务（工具/错误恢复/记忆/循环保护/多步）
├── BENCHMARKS.md          真实模型实测台账（消融 / 双层对照 / 边界 / 复现命令）
├── examples/run_mock.py   四段离线演示（v1 模块逐个验）
├── examples/run_super.py  单 loop vs 双层路由对照实验（支持 --online）
├── examples/run_async.py  异步提交 + 换进程读状态 + 断点续跑演示
├── examples/check_online.py  连通性自检（chat + embeddings，确认 key/端点）
├── tests/test_smoke.py    40 个单测（v1）
├── tests/test_super.py    23 个单测（filter / watchdog / router / supervisor）
├── tests/test_async_embed.py  20 个单测（远端 embedding / 熔断 / 维度守卫 / 异步 Worker / 续跑）
└── tests/test_p0_hardening.py 9 个单测（预算配额 / 中文裁剪 / ref 逃逸 / 沙箱 / checkpoint 并发）
```

## 6. 设计决策与取舍

- **零依赖**：只用 stdlib。理由：harness 的难点在**上下文和状态管理**，不在 HTTP 客户端；去掉依赖后任何机器 `python run_mock.py` 就能验证，任何人都能当场跑。
- **哈希 embedding 不是终局，而是兜底**：`HashingEmbedder`（CJK 2-gram + ascii 词 + crc32 → 512 维 L2 归一化）默认跑，配了 `CH_EMBEDDING_MODEL` 就走真实向量。真正值得讲的是**降级路径**：远端 → 熔断 → 本地，任何一环挂了任务都还能跑完，只是判得糙一点。
- **embedding 熔断是整场生效**：一次失败就整场降级，不做逐次重试（否则每条消息都要等一次 30 秒超时）；降级后靠**维度守卫**跳过长度不匹配的历史向量，避免点积静默算错。
- **存算分离拆成两层**：`workers` 表只存元信息（状态/答案/成本），执行进度留在 `checkpoint.runs`，run_id 复用 worker_id。都存进度就会互相对不上。
- **Watchdog 的停滞签名要含输出**：只看工具名会把「失败一次后重试」和「读不同文件」误判成原地打转——必须 `name+args+输出摘要` 全同才算停滞。这个 bug 当场把评测通过率从 100% 打到 80%，就是靠数字发现的。
- **跑偏判定要认硬锚点**：真实模型会用一句话确认「已更新为 2500 token」，字数少、与问题重合度低，纯相似度会把它判成跑偏。规则是：短于 40 字的确认类文本不判跑偏；文本命中任务里的数字/标识符（`2500`、`api_client.py`）就不算跑偏。
- **确定性 DAG 而非自主规划**：任务边界清晰时，确定性编排的 token 成本和可复现性都远好于让模型自由规划；模型只负责节点内的动作。等有评估数据后再考虑 planner。
- **工具一律不抛异常**：白名单/参数校验/审批/内部异常全部转成结构化 `ToolResult` 回灌给模型，让它自己修正参数——这是「错误恢复」能力的基础，抛异常等于放弃一次纠错机会。
- **裁剪按组**：`(assistant(tool_calls) + tool 结果)` 必须同生同死，否则会出现孤立的 tool 消息，多数 API 会直接 400。
- **目标重注入**：长任务里目标会被历史稀释，每 4 步追加一次【目标提醒】，比调大 system prompt 有效且便宜。
- **灰区 fail-open**：过滤器只有拿不准时才调模型，且拿不到判断就放行。理由是不对称代价——漏掉一条真实任务的损失远大于多处理一条噪音。
- **Watchdog 不调模型**：它是成本控制器，用模型监控模型既烧钱又失去独立性（出问题时最可疑的就是模型）。
- **背压分两个数**：`max_workers` 管「同时跑几个」，`queue_factor` 管「还能接多少」。混成一个数就会出现「一忙就丢任务」。
- **Router 自带 Worker 创建**：`TaskRouter.route()` 返回带 `worker_id` 的决策，Supervisor 只负责跑——这样「决策」和「执行」可以分开测。

## 7. 已知限制（主动说出来比被问出来好）

- 工具串行执行，未做并行/并发限制流（分布式并发控制是下一步）。Worker 之间并行，Worker 内的工具调用仍然串行。
- 无语义缓存，同一问题重复调用仍消耗 token。
- MCP 只实现 tools/list + tools/call，未覆盖 resources/prompts/sampling。
- `ScriptedLLM` 的离线指标不能代表效果；真实对照需要 `--online`。
- 没有 Web UI / 多租户 / 鉴权。
- 默认用字符哈希 embedding，同义改写（「登录挂了」vs「认证失败」）判不出；要开 `CH_EMBEDDING_MODEL` 并配一个支持 `/embeddings` 的端点。
- 异步 Worker 是**单进程多线程**，不是分布式；进程崩了靠 checkpoint 续跑，不是靠副本存活。
- 熔断之后没有恢复探测（没有半开状态），一场跑完就一直是降级态。
- 任务集只有 10 条，指标方差大；扩到 40 条并按步数分桶后才适合讲「能力提升」。

## 8. 下一步（roadmap）

1. tool 调用并行化 + 每工具超时与熔断（对应「并发控制」方向）
2. 语义缓存 + 小模型分流阈值自适应（对应「降本 70%」方向）
3. ~~Router 话题相似度换成真实 embedding~~（v2.1 已完成，`CH_EMBEDDING_MODEL` 开启）
4. ~~Worker 异步执行 + 存算分离~~（v2.1 已完成，`worker_store.py` + `checkpoint`）
5. embedding 熔断加半开恢复探测（现在一崩就整场降级）
6. MCP resources/prompts + 远端 SSE 传输
7. 长任务评估集扩到 40 条，按「步数分桶」看成功率（暴露 >15 步衰减）

## 9. 接真实模型（跑出你自己的数字）

```bash
cp .env.example .env        # 填 CH_BASE_URL / CH_API_KEY / CH_MODEL
# 想要真实 embedding 再填三行（chat 端点没有 /embeddings 时单独指向别的厂商）：
#   CH_EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
#   CH_EMBEDDING_API_KEY=sk-xxx
#   CH_EMBEDDING_MODEL=BAAI/bge-m3

# ① 连通性自检：一次 chat + 一次 embeddings，确认 key 和端点没问题
python examples/check_online.py

# ② v1 消融：上下文预算 + 三层记忆的收益
python -m ch.evaluate --variants baseline budget full --online --out .artifacts/real-eval.json

# ③ v2 对照：单 loop 串行 vs Router–Worker 双层
python examples/run_super.py --online

# ④ 异步 + 存算分离（不需要模型，纯本地）
python examples/run_async.py
```

`.env` 支持任意 OpenAI 兼容端点（DeepSeek / 通义 / vLLM / Ollama 均可），代码按 `price_per_mtok` 表算成本，接新模型只需在 `src/ch/config.py` 加一行。

没配 `.env` 时 `--online` 会明确提示并以退出码 2 结束，不会悄悄跑回离线脚本——**避免把 mock 数字当成真实结果对外引用**。

**已知实测结果与口径**：见 [BENCHMARKS.md](BENCHMARKS.md)。用 DeepSeek 跑时要注意——它**没有 `/embeddings` 端点**，配了 `CH_EMBEDDING_MODEL` 也会 404 然后熔断到本地哈希向量（这是设计好的降级路径，评测照常跑完）。

`tasks/tasks.json` 只有 10 条，想要更有说服力就扩到 40 条（roadmap 第 7 条）。
