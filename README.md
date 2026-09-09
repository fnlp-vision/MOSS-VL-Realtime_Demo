# MOSS-VL Realtime Demo

English | [简体中文](./README_zh.md)

A browser application for realtime video and voice interaction with MOSS-VL, with optional speech recognition, speech synthesis, and long-term memory.

## Related Projects

| Project | Role |
| --- | --- |
| [MOSS-VL-Realtime-SGLANG](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime-SGLANG) | Model weights, configuration, tokenizer, and processor |
| [sglang-omni-realtime](https://github.com/fnlp-vision/sglang-omni-realtime) | Stateful streaming inference and GPU scheduling |
| This repository | Browser UI, ASR/TTS, memory, and REST/WebSocket gateway |

Start here for the complete application. For a standalone model service, use the [backend README](https://github.com/fnlp-vision/sglang-omni-realtime#readme).

## Features

- Camera, screen, image, and video-file input.
- Streaming captions, text and voice questions, and response interruption.
- Optional SenseVoice ASR, local MOSS-TTS-Nano, and text/image memory.
- Multi-session backend access, context monitoring, and memory rollover.
- REST session management and a WebSocket gateway for external clients.

## Quick Start

Prerequisites: Linux x86_64, Python 3.12, a CUDA 13-compatible NVIDIA driver, Git, a C/C++ compiler, CMake, and FFmpeg. The installer manages application dependencies, not system packages or drivers.

From the repository root:

```bash
bash bootstrap.sh --doctor-only
bash bootstrap.sh --with-memory --with-asr --with-tts
.venv/bin/python scripts/repro/run.py up --main-gpu 0 --memory-gpu 1
```

The installer prepares Python/Node tools, separate Demo and backend environments, npm dependencies, the backend source, and models. The full configuration uses two separate GPUs. Omit all `--with-*` options for video/text only on one GPU. Existing unmanaged `.venv` directories are not overwritten.

For a source bundle containing both `demo/` and `backend/`, replace the installation command with:

```bash
bash bootstrap.sh --backend-source ../backend --with-memory --with-asr --with-tts
```

Open **http://localhost:18502**, switch to video-call mode, choose a media source, and connect. Voice input defaults to push-to-talk; select VAD for automatic capture. Remote camera/microphone access requires HTTPS. This configuration does not enable offline chat or require cloud TTS credentials.

```bash
.venv/bin/python scripts/repro/run.py status
.venv/bin/python scripts/repro/smoke.py
.venv/bin/python scripts/repro/run.py down
```

Wait for model loading and kernel compilation on first startup. Use `--base-port 19500` to move the port group; the browser port becomes `19502`.

## Configuration

| Component | Default |
| --- | --- |
| Main VLM | GPU 0; 4 sessions; 131072 context; memory fraction 0.5 |
| Memory 4B | GPU 1; 4 concurrent requests; 16384 context; 65536 KV tokens; memory fraction 0.2 |
| ASR / local TTS / retrieval models | CPU |
| Backend / API / browser | 18500 / 18501 / 18502 |
| pi-agent / 4B / TTS | 18503 / 18504 / 18505 |

Adjust memory and concurrency for your hardware. See the [installation guide](./deployment/repro/README.md) and [compatibility notes](./docs/compatibility.md) (Chinese) for configuration details.

## API and Deployment

```text
Browser -> /api/session/{id}/ws -> orchestration -> backend /v1/video/realtime
External client -> gateway /v1/realtime -> backend /v1/video/realtime
```

The browser and backend protocols are different. The web server proxies `/api` only; external gateway clients use the API port or an explicitly configured `/v1` reverse proxy. Browser and gateway pools do not share a global admission counter.

Services bind to loopback by default. Public deployments must add authentication, session ownership checks, TLS, and rate limits. A one-time WebSocket token is not a complete authentication system.

## Documentation

- [Gateway protocol](./docs/gateway_contract.md) and [operations](./docs/ops_runbook.md).
- [Memory service](./services/pi_agent/README.md) and [capacity planning](./docs/vlm_memory_capacity.md).
- [Manual/legacy deployment](./docs/manual_deployment.md) and [deployment operations](./docs/deployment_operations.md). Keep legacy HF/NPU environments separate from the recommended setup.

## Development

Install `requirements-dev.txt` in a separate development environment, then run:

```bash
python scripts/run_tests.py
python scripts/dev/check_env.py --check
npm run build
```

The test runner supports both script-based suites and pytest. Use it rather than collecting every suite with a single pytest invocation.

## License

The model, backend, and vendored components retain their respective licenses. See [OpenMOSS/MOSS-VL](https://github.com/OpenMOSS/MOSS-VL) for the model project and [SGLang-Omni](https://github.com/sgl-project/sglang-omni) for the upstream inference framework.
