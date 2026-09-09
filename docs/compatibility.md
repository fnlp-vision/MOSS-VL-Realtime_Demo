# 组件兼容性

推荐配置通过独立环境连接 Demo 与实时推理后端，不在同一个 Python 环境混装两者依赖。

| 组件 | 配套版本 |
| --- | --- |
| 主模型 | `OpenMOSS-Team/MOSS-VL-Realtime-SGLANG` |
| 后端 | `fnlp-vision/sglang-omni-realtime` |
| 后端 Python / Torch / Transformers | 3.12 / 2.11.0 / 5.12.1 |
| SGLang / FlashInfer | 0.5.16 / 0.6.14 |
| CUDA 编译器、CRT、NVVM | 13.0.88，配合 CUDA 13.0 运行库 |
| Demo Python / Torch / Transformers | 3.12 / 2.8.0 CPU / 4.57.1 |
| Node | 22.12.0 |

安装器自动选择对应源码和模型版本，详细依赖保存在 `deployment/repro`。使用包含两个仓库的源码包时，通过 `--backend-source ../backend` 安装配套后端。

## 模型与应用边界

- Demo 使用 HTTP/WebSocket 调用后端，不在其 CPU 环境加载 Transformers 5.12.1 模型。
- 原版 `MOSS-VL-Realtime` 的 Transformers 4.57 自定义代码不可替换本兼容模型的文件。
- 模型支持的 256K context 不代表部署为每个会话分配 256K；推荐服务配置使用 131072。
- ASR、TTS、memory 和跨 context rollover 由 Demo 提供，不属于模型权重的内置能力。
- 模型的离线 Python API 与 Demo 是否启用离线聊天是两回事。

升级代码、模型或依赖时应一起检查兼容性。HF worker、NPU 和其他语音引擎使用各自环境与部署说明。
