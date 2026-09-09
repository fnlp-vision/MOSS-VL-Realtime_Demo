# MOSS-VL Realtime Demo

[English](./README.md) | 简体中文

MOSS-VL 实时视频与语音交互应用，提供浏览器界面，以及可选语音识别、语音合成和长期记忆。

## 配套项目

| 项目 | 职责 |
| --- | --- |
| [MOSS-VL-Realtime-SGLANG](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime-SGLANG) | 模型权重、配置、tokenizer 和 processor |
| [sglang-omni-realtime](https://github.com/fnlp-vision/sglang-omni-realtime) | 持续会话、流式推理及 GPU 调度 |
| 本仓库 | 浏览器界面、ASR/TTS、memory 和 REST/WebSocket 网关 |

体验完整应用，从本页开始；只部署模型服务，使用[后端 README](https://github.com/fnlp-vision/sglang-omni-realtime/blob/main/README_zh.md)。

## 功能

- 摄像头、屏幕、图片和视频文件输入。
- 流式字幕、文字及语音提问、回答打断。
- 可选 SenseVoice ASR、MOSS-TTS-Nano 本地语音及文本/图像 memory。
- 多会话后端接入、上下文观测与 memory rollover。
- 面向外部客户端的 REST 会话管理和 WebSocket 网关。

## 快速开始

需要 Linux x86_64、Python 3.12、兼容 CUDA 13 的 NVIDIA 驱动、Git、C/C++ 编译器、CMake 和 FFmpeg。安装器负责应用依赖，不安装系统软件包或驱动。

在仓库根目录执行：

```bash
bash bootstrap.sh --doctor-only
bash bootstrap.sh --with-memory --with-asr --with-tts
.venv/bin/python scripts/repro/run.py up --main-gpu 0 --memory-gpu 1
```

安装器自动准备 Python/Node 工具、独立的 Demo 与后端环境、npm 依赖、后端源码和模型。完整配置使用两张独立 GPU；不带 `--with-*` 只安装视频/文本配置，仅使用一张主 GPU。已有的非托管 `.venv` 不会被覆盖。

使用同时包含 `demo/` 和 `backend/` 的源码包时，安装命令改为：

```bash
bash bootstrap.sh --backend-source ../backend --with-memory --with-asr --with-tts
```

浏览器打开 **http://localhost:18502**，切换到视频通话，选择媒体源并连接。语音默认按住说话，自动收音可选 VAD。远程摄像头与麦克风需要 HTTPS。当前配置不启用离线聊天，也不需要云端 TTS 凭据。

```bash
.venv/bin/python scripts/repro/run.py status
.venv/bin/python scripts/repro/smoke.py
.venv/bin/python scripts/repro/run.py down
```

首次启动需要加载模型和编译内核，请等待就绪。端口冲突时使用 `--base-port 19500` 切换整组端口，网页相应为 `19502`。

## 配置

| 组件 | 默认值 |
| --- | --- |
| 主 VLM | GPU 0；4 会话；131072 context；显存比例 0.5 |
| Memory 4B | GPU 1；4 并发；16384 context；65536 KV token；显存比例 0.2 |
| ASR / 本地 TTS / 检索模型 | CPU |
| 后端 / API / 网页 | 18500 / 18501 / 18502 |
| pi-agent / 4B / TTS | 18503 / 18504 / 18505 |

请按硬件调整显存和并发参数，详见[安装指南](./deployment/repro/README.md)和[兼容说明](./docs/compatibility.md)。

## 接口与部署

```text
浏览器 -> /api/session/{id}/ws -> 会话编排 -> 后端 /v1/video/realtime
外部客户端 -> 网关 /v1/realtime -> 后端 /v1/video/realtime
```

浏览器协议与底层模型协议不同。网页仅代理 `/api`；外部网关客户端访问 API 端口或显式配置 `/v1` 反向代理。浏览器与网关的实例池不共享全局准入计数。

服务默认绑定 loopback。公开部署必须补充鉴权、会话归属校验、TLS 和限流；一次性 WebSocket token 不等同于完整鉴权。

## 更多文档

- [网关协议](./docs/gateway_contract.md)与[运维手册](./docs/ops_runbook.md)。
- [Memory 服务](./services/pi_agent/README.md)与[容量规划](./docs/vlm_memory_capacity.md)。
- [手工与历史部署](./docs/manual_deployment.md)、[部署运维](./docs/deployment_operations.md)。旧版 HF/NPU 环境与推荐配置分开使用。

## 开发

在独立开发环境安装 `requirements-dev.txt` 后执行：

```bash
python scripts/run_tests.py
python scripts/dev/check_env.py --check
npm run build
```

测试入口同时支持脚本式套件和 pytest，不应直接用一次 pytest 调用替代全部测试入口。

## 许可证

模型、后端和 vendored 组件分别遵循各自许可证。模型项目见 [OpenMOSS/MOSS-VL](https://github.com/OpenMOSS/MOSS-VL)，推理框架上游见 [SGLang-Omni](https://github.com/sgl-project/sglang-omni)。
