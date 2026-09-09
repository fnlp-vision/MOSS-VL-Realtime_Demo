# pi_agent — Loopback HTTP 决策/压缩服务

本服务随 Demo 仓库的 `services/pi_agent` 目录交付，不再依赖后端旁的未跟踪目录。
推荐使用仓库根目录的 `bootstrap.sh --with-memory` 安装固定 npm 依赖。
本地无鉴权部署显式设置 `AIGW_AUTH_MODE=local`，仅允许 loopback 端点，不读取内部 key 文件。
其他部署必须显式提供 `AIGW_API_KEY` 或 `AIGW_KEY_FILE`；仓库不提供默认私有凭据路径。

基于 `@earendil-works/pi-agent-core@0.74.2` 和 `@earendil-works/pi-ai@0.74.2` 的本地回环
HTTP 服务（Node 22，纯 ESM `.mjs`，无 TS 构建链路），供 board Python 侧通过 HTTP 调用：

- `POST /decide`：判定回答 pending 用户消息前是否需要长期记忆检索，并给出检索 query。
- `POST /compact`：把带时间戳的会话 journal 压缩成 `summary` + `pins`。
- `GET /health` 或 `/ready`：检查实际模型生成、模型身份和分词预算接口；未就绪返回 503。
- `GET /live`：仅检查本进程存活。

当前部署使用本地 SGLang 的 Qwen3-4B-Instruct-2507。预算检查要求后端同时提供
`/v1/tokenize`（支持 messages 和 chat template）和 `/get_server_info`（实际上下文上限）。
未提供这两个接口的通用 OpenAI-compatible 网关不能直接替代；计数失败会拒绝调用，不回退到字符估算。

## 文件

| 文件 | 职责 |
|---|---|
| `service.mjs` | HTTP 服务入口：env 引导（no_proxy、key 加载）、路由、错误统一为 502 |
| `decide.mjs` | `/decide` 实现：hop（单次结构化调用）/ agent（Agent + 工具 loop）双模式 |
| `compact.mjs` | `/compact` 实现 |
| `prompts.mjs` | 全部 LLM prompt 模板 |
| `aigw.mjs` | AIGW 客户端封装：pi-ai `completeSimple`、key 读取、严格 JSON 解析与一次重试 |

## 依赖

仅用 `node_modules` 中已装的 `@earendil-works/pi-agent-core` / `@earendil-works/pi-ai` 和
Node 内置 `http`/`fs`/`fetch`，不新增 npm 依赖。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `PI_PORT` | `38080` | 监听端口（固定绑定 127.0.0.1） |
| `PI_AGENT_MODE` | `hop` | `/decide` 模式：`hop` 单次调用；`agent` 完整 agent loop |
| `BOARD_MEMORY_URL` | `http://127.0.0.1:8081` | board 记忆服务地址（agent 模式工具回调用） |
| `AIGW_BASE_URL` | `http://127.0.0.1:38090/v1` | SGLang-compatible base URL |
| `AIGW_KEY_FILE` | 空 | 显式指定的 API key 文件（启动时读取，不打印） |
| `AIGW_AUTH_MODE` | `token` | `local` 允许无 key 的本地回环服务；默认模式需显式凭据 |
| `AIGW_API_KEY` | 空 | 直接给 key（优先级高于 key 文件） |
| `AIGW_MODEL` | `Qwen3-4B-Instruct-2507` | decide / agent 路由模型 |
| `AIGW_COMPACT_MODEL` | 同 `AIGW_MODEL` | compact 路由模型；必须与实际后端匹配 |
| `PI_DECIDE_TIMEOUT_MS` | `20000` | hop 模式 LLM 请求超时 |
| `PI_AGENT_TIMEOUT_MS` | `45000` | agent 模式整轮超时（超时即 abort） |
| `PI_COMPACT_TIMEOUT_MS` | `60000` | compact 超时上限 60s |
| `PI_CONTEXT_TOKENS` | `16384` | 应用上下文上限，与后端实际上限取较小值，不改变 GPU 预分配 |
| `PI_COMPACT_CHUNK_TOKENS` | `4096` | 单段最终 system + user 输入预算，按完整 QA 分段 |

**部署必须保证 `no_proxy` 覆盖 `aigw.sotatts.online`**，否则进程经 HTTP(S) 代理出站会
TLS 失败。服务启动时会自动把该 host 追加进 `process.env.no_proxy`/`NO_PROXY` 兜底，
但仍建议在部署环境的 shell/systemd unit 里显式配置：

```bash
export no_proxy="${no_proxy:+$no_proxy,}aigw.sotatts.online"
export NO_PROXY="$no_proxy"
```

## 启动

```bash
cd pi_agent
export AIGW_DECIDE_BASE_URL=http://127.0.0.1:38090/v1
export AIGW_COMPACT_BASE_URL=http://127.0.0.1:38090/v1
export AIGW_DECIDE_MODEL=Qwen3-4B-Instruct-2507
export AIGW_COMPACT_MODEL=Qwen3-4B-Instruct-2507
export AIGW_LOCAL_NO_REASONING=1
export AIGW_AUTH_MODE=local
npm start                 # = node service.mjs（默认 hop 模式，127.0.0.1:38080）

PI_AGENT_MODE=agent npm start          # agent 模式（需要 board memory 在线）
```

## 契约示例

```bash
# 健康检查
curl -s http://127.0.0.1:38080/health
# {"ok":true,"model":"kimi-k3","compact_model":"kimi-k3","mode":"hop"}

# 决策
curl -s http://127.0.0.1:38080/decide -H 'content-type: application/json' -d '{
  "conversation_id": "cam-42",
  "recent_turns": "用户: 晚饭吃什么好？\n助手: 要不来点清淡的，比如粥。",
  "pending_user_text": "我之前给你看的那台相机是什么牌子？"
}'
# 200 {"retrieve":true,"query":"...","reason":"..."}

# 压缩
curl -s http://127.0.0.1:38080/compact -H 'content-type: application/json' -d '{
  "conversation_id": "cam-42",
  "journal": "用户: ...\n助手: ...\n用户: ...",
  "summary_max_tokens": 200
}'
# 200 {"summary":"...","pins":["..."]}
```

模型失败、输出校验失败或计数接口不可用返回 `502 {"error": "..."}`；输入预算超限返回 413；请求体不是 JSON 对象返回 400；
未知路由返回 404。

调用方可用 `X-Memory-Timeout-Ms` 传入剩余总期限；实际期限取请求值与本端配置的较小值。
期限覆盖分词、生成、JSON 修复和分段压缩，超时返回 504；客户端断开时取消下游请求。
SDK 内部自动重试关闭，避免与 Demo 重试叠加。`/ready` 同时返回生效配置和实现指纹，
供部署脚本识别旧代码/旧参数；指纹不是模型版本。

## 预算与完整性

- 每次生成前校验最终 payload，包括 system、近期对话、当前问题、工具定义与工具结果；JSON 重试也重新计数。保留生成额度和 256 token 安全余量。
- `/compact` 按完整 QA 分段，单次作业从全部原始 journal 重新生成，段间合并摘要和 pins；不截掉末尾新记录。最多 32 段；单个 QA 超限或整份作业失败时不返回部分摘要。
- `summary_max_tokens` 范围 16–200，默认 200；summary 同时不超过 200 字符。pins 最多 16 条，每条最多 256 字符、合计最多 320 token，必须逐字来自原文。超出总预算时按重要性顺序保留整条 pin 并记录裁减日志，不截断字符串。Demo 组装时仍受置顶层总预算约束。
- `retrieve` 必须为 JSON 布尔值；`query` 为字符串或 null；query/reason 最多 512 字符。非法输出不会作为有效决策使用。
- 原始 journal 保留在 Demo 数据库；失败时 Demo 明确记录降级日志，使用近期原文，不将降级标记为压缩成功。

回归测试：`node --test budget.test.mjs`（使用本机临时模拟后端，无需 GPU）。

## 双模式说明

- **hop**（默认，`PI_AGENT_MODE=hop`）：单次结构化 LLM 调用。系统 prompt 要求严格 JSON
  `{retrieve, query, reason}`；解析失败自动加一条"上次不是合法 JSON"的 nudge 重试一次，
  仍失败则 502。实测延迟 ~1.5–2.2s。
- **agent**（`PI_AGENT_MODE=agent`）：用 pi-agent-core 的 `Agent` 跑完整 agent loop，
  注册 `memory_retrieve` 工具（0.74.2 没有独立 `registerTool` API，通过
  `agent.state.tools` 注册）。工具的 `execute` 回调 HTTP 调用 board 的
  `POST {BOARD_MEMORY_URL}/api/memory/retrieve`，body
  `{"conversation_id","query","top_k":4}`，结果作为 tool result 回到模型，再由模型输出
  最终 JSON 结论。board memory 不在线时工具调用报错，agent 会带着错误信息继续给出结论
  （倾向 retrieve=true），不会整体 502。实测一轮工具调用延迟 ~12s，故有独立
  `PI_AGENT_TIMEOUT_MS`（默认 45s）。

## 已知限制

- 服务无鉴权，仅监听 127.0.0.1，切勿暴露到外部。
- 并发请求共享同一个 AIGW key，无并发/限流控制。
- hop 模式的严格 JSON 依赖模型遵循指令；解析失败只有一次重试机会。
- agent 模式端到端验证使用了 board memory 的本地 stub（真实 board 联调待其 8081 服务
  上线后进行）。
- AIGW 上 `kimi-k3` 的 thinking 内容（pi-ai 中的 thinking block）原样透传展示，不影响
  最终 text 输出解析。
