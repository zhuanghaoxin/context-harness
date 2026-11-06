# BENCHMARKS：真实模型实测数字

本文件记录的是**真实 API 调用**跑出来的数字，不是 `ScriptedLLM` 回放。
对外引用数字时，引这里的表；`README.md` 里那套离线数字只用来验证机制正确性。

- 跑的时间：2026-11-04
- 模型：`deepseek-chat`（价格表按 `src/ch/config.py` 里 `price_per_mtok = (0.27, 1.10)` 美元/百万 token 折算）
- 端点：`https://api.deepseek.com/v1`（OpenAI 兼容）
- 机器：本机单进程，Python 3.10.2，Windows
- 重复次数：每个配置跑 **3 次**，下表是均值，括号里是三次的最小-最大值
- 原始明细：`.artifacts/real-eval.json`、`.artifacts/real-eval-1.json`、`.artifacts/real-eval-2.json`（含每条任务的回答与停机原因）

## 1. 三档消融（10 条自建任务）

```powershell
python -m ch.evaluate --variants baseline budget full --online --out .artifacts/real-eval.json
```

| variant | 上下文预算 | 记忆 | 通过率 | token/任务 | 成本/任务 | 平均步数 |
|---|---|---|---|---|---|---|
| `baseline` | 200000（不裁） | 无 | 100% | **3237**（2650–3956） | **$0.00102** | 2.30 |
| `budget` | 2500 | 无 | 100% | 3537（3458–3666） | $0.00112 | 2.73 |
| `full` | 2500 | 三层记忆 | 100% | **2482**（2312–2627） | **$0.00079** | 2.23 |

**净收益（full 相对 baseline）**：token/任务 **-23.3%**，成本/任务 **-22.9%**，通过率持平（本来就都满分）。

### 值得主动说出来的反例：只开裁剪比不裁更贵

`budget` 相对 `baseline`：token **+9.3%**，成本 **+8.9%**，平均步数 2.30 → 2.73。

原因：上下文一被裁，模型看不到先前拿到的工具结果，只能重新调用一次工具把信息取回来——省下的 prompt token 被多出来的 step 吃掉了。这说明**「省 token」本身不是目标，单位任务成本才是**；预算裁剪必须和记忆（把结论落成结构化事实，不必重新取）一起开才有净收益，这也正是 `full` 能到 -23% 的原因。

这条反例比 -23% 更值钱：它证明消融是「测出来的」而不是「调出来的」。

## 2. 单 loop 串行 vs Router–Worker 双层

```powershell
python examples/run_super.py --online
```

10 条输入 = 6 条真实任务 + 4 条噪音（打招呼/心跳类）。同一批输入、同一套 checker。

| 指标 | single（单 loop） | super（双层） | 变化 |
|---|---|---|---|
| 实跑 loop | 10 | 6 | 4 条噪音 0 成本拒收 |
| 完成 | 10 | 6 | — |
| 通过率 | 100% | 100% | — |
| makespan | 21967 ms | **6639 ms** | **-69.8%** |
| P95 端到端延迟 | 21403 ms | **6461 ms** | **-69.8%** |
| 峰值并发 | 1 | 4 | — |
| 主动委派 | 0 | 6 | 用户没要求拆分，Router 自己分的工 |
| 总 token | 5711 | 3944 | -30.9% |
| 总成本 | $0.00303 | $0.00238 | -21.4% |

三次单跑明细：

| run | single makespan | super makespan | single token | super token |
|---|---|---|---|---|
| 1 | 23043 ms | 7301 ms | 5862 | 4239 |
| 2 | 19504 ms | 6984 ms | 5563 | 3888 |
| 3 | 23353 ms | 5632 ms | 5709 | 3705 |

## 3. 这些数字的边界（被质疑时先自己说）

1. **任务集只有 10 条**，单任务 token 波动 2650–3956，run-to-run 方差约 ±20%。所以结论是**机制级**（过滤/并行/裁剪确实在这条链路上省了东西），不是**能力级**（不代表模型在真实业务上也能涨）。想变成能力级证据，要把任务集扩到 40 条并按步数分桶。
2. **token 收益主要来自「不跑噪音」和「不重复取信息」**，不是模型变聪明了。`super` 的 30.9% 里，4 条噪音在 single 里被完整跑了 4 个 loop。
3. **makespan 收益来自并发度**（1 → 4），单条任务的绝对耗时没有变快；延迟长尾也随之下移，所以 P95 和 makespan 同幅下降。
4. **embedding 是降级态**：`CH_EMBEDDING_MODEL=text-embedding-3-small` 填了，但 DeepSeek 的端点没有 `/embeddings`（实测 `HTTP 404`），`ResilientEmbedder` 按设计整场熔断到本地哈希向量。也就是说上面所有数字都是在**降级态**下取得的——外部依赖挂了评测仍能跑完，只是话题相似度判得糙。要真用向量，换一个有 `/embeddings` 的端点（通义、硅基流动、OpenAI）。
5. **Super-agent 原论文的 P95 15.7min vs 29.5min（-47%）不能和这里的 -69.8% 直接比**：口径不同（他们 300 条消息 / 20 项真实工作请求，我 10 条输入）。引用别人的数字时要说明是别人的。

## 4. 复现命令

```powershell
cd context-harness
$env:PYTHONPATH='src'

# 连接性自检（一次 chat + 一次 embeddings，确认 key/端点）
python examples/check_online.py

# 消融（约 $0.01/次）
python -m ch.evaluate --variants baseline budget full --online --out .artifacts/real-eval.json

# 双层对照（约 $0.006/次）
python examples/run_super.py --online

# 离线：不花钱，验证机制
python examples/run_mock.py
python examples/run_async.py
python -m unittest discover -s tests -t .      # 105 个单测
```

## 5. 跑数字过程中被数字抓出来的 bug

这些都是「跑真实模型之前看不出来」的：

1. **Watchdog 误杀正常重试**：停滞判定只看工具名，连续 3 次 `read_file` 即判死循环，把「失败后重试」「读不同文件」也算进去 → 通过率 100% → 80%（`stop_reasons={'final': 8, 'watchdog:loop': 2}`）。修法：停滞签名改成 `工具名 + 参数 + 输出摘要` 全同。
2. **Watchdog 误杀短确认**：真实模型对「把预算改成 2500 token」回「已更新为 2500 token」，字符 2-gram Jaccard 只有 0.02 → 判 off_topic 停机，通过率 100% → 90%。修法：① 短于 40 字的确认类文本不判跑偏；② 命中任务里的硬锚点（数字/标识符，如 `2500`、`api_client.py`）不算跑偏。
3. **单测被本地 `.env` 污染**：`Config.from_env()` 在填了真实 key/embedding 之后，单测会去真调远端，首个 submit 被拖到 774ms，直接把「并行更快」这条断言测成失败。修法：测试用例里显式钉死 `base_url/api_key/embedding_model`。
