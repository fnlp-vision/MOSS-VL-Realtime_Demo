"""OpenAI-compatible surface (v1/chat/completions + v1/models) over the Realtime VLM.

Purpose: let standard OpenAI clients / benchmark tools (evalscope perf,
vllm_benchmark, curl) drive the in-process Realtime MOSS-VL without knowing the
gateway's native /api/chat/stream protocol. Requests are translated into a
native ChatRequest and streamed back as OpenAI SSE chunks.

Only text + image messages are supported (images as data-URL / base64 in
content parts, matching the OpenAI shape). Video parts and conversation
recording are not exposed here.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator, List, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from ..deps import Runtime, get_runtime
from ..logging_conf import get_logger
from ..schemas import ChatMessage, ChatRequest, GenerationParams

log = get_logger(__name__)
router = APIRouter(prefix="/v1", tags=["openai"])


def _model_id(rt: Runtime) -> str:
    return getattr(getattr(rt, "vlm", None), "model_path", "MOSS-VL-Realtime") or "MOSS-VL-Realtime"


# ----------------------------- translation -----------------------------


def _translate_messages(messages: List[dict]) -> List[ChatMessage]:
    """OpenAI messages → native ChatMessage (content list or string).

    OpenAI image parts use {"type": "image_url", "image_url": {"url": …}};
    the native shape uses {"type": "image", "media": …}. Only the LAST user
    message may carry images (matches the native _prepare_chat_messages rule
    that media attaches to the last user turn).
    """
    out: List[ChatMessage] = []
    for m in messages:
        role = str(m.get("role", "user"))
        content = m.get("content")
        if isinstance(content, str):
            out.append(ChatMessage(role=role, content=content))
            continue
        if not isinstance(content, list):
            out.append(ChatMessage(role=role, content=str(content or "")))
            continue
        parts: List[dict] = []
        for part in content:
            if not isinstance(part, dict):
                parts.append({"type": "text", "text": str(part)})
                continue
            ptype = part.get("type")
            if ptype in ("image_url", "image"):
                url = part.get("image_url") or {}
                if isinstance(url, dict):
                    url = url.get("url", "")
                url = url or str(part.get("image") or "")
                parts.append({"type": "image", "media": str(url)})
            elif ptype == "text":
                parts.append({"type": "text", "text": str(part.get("text") or "")})
            # unknown part types are dropped (mirrors the native adapter)
        out.append(ChatMessage(role=role, content=parts))
    return out


def _translate_params(body: dict) -> GenerationParams:
    p = GenerationParams()
    if body.get("max_tokens") is not None:
        p.max_new_tokens = int(body["max_tokens"])
    if body.get("temperature") is not None:
        p.temperature = float(body["temperature"])
    if body.get("top_p") is not None:
        p.top_p = float(body["top_p"])
    if body.get("top_k") is not None:
        p.top_k = int(body["top_k"])
    if body.get("repetition_penalty") is not None:
        p.repetition_penalty = float(body["repetition_penalty"])
    if body.get("do_sample") is not None:
        p.do_sample = bool(body["do_sample"])
    return p


# ----------------------------- SSE shaping -----------------------------


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _chunk(chat_id: str, model: str, delta: dict, finish: Optional[str] = None) -> str:
    return _sse({
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    })


async def _chat_completions_sse(rt: Runtime, body: dict) -> AsyncIterator[str]:
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    model = body.get("model") or _model_id(rt)
    messages = _translate_messages(body.get("messages") or [])
    if not messages:
        raise ValueError("messages is required")

    req = ChatRequest(messages=messages, params=_translate_params(body), use_template=True)
    vlm = rt.vlm
    if vlm is None or not getattr(vlm, "is_loaded", lambda: False)():
        raise RuntimeError("Realtime VLM not loaded")

    text_parts: List[str] = []

    yield _chunk(chat_id, model, {"role": "assistant", "content": ""})

    # generate_stream yields per-token text deltas
    generator = vlm.generate_stream(req)
    try:
        async for delta in generator:
            text_parts.append(delta)
            if delta:
                yield _chunk(chat_id, model, {"content": delta})
                yield "\n\n"
    finally:
        if hasattr(generator, "aclose"):
            await generator.aclose()

    yield _chunk(chat_id, model, {}, finish="stop")
    yield "\n\n"
    # final usage-only chunk (include_usage style)
    content = "".join(text_parts)
    yield _sse({
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": _estimate_prompt_tokens(vlm, body.get("messages") or []),
            "completion_tokens": _estimate_completion_tokens(vlm, content),
            "total_tokens": 0,
        },
    })
    yield "\n\n"
    yield "data: [DONE]\n\n"


def _estimate_prompt_tokens(vlm: Any, messages: List[dict]) -> int:
    try:
        tokenizer = getattr(getattr(vlm, "processor", None), "tokenizer", None)
        if tokenizer is None:
            return 0
        text = ""
        for m in messages:
            c = m.get("content")
            if isinstance(c, str):
                text += c + "\n"
            elif isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text += str(part.get("text", "")) + "\n"
        return len(tokenizer.encode(text))
    except Exception:  # noqa: BLE001
        return 0


def _estimate_completion_tokens(vlm: Any, text: str) -> int:
    try:
        tokenizer = getattr(getattr(vlm, "processor", None), "tokenizer", None)
        if tokenizer is None:
            return 0
        return len(tokenizer.encode(text))
    except Exception:  # noqa: BLE001
        return 0


# ----------------------------- routes -----------------------------


@router.get("/models")
def list_models(rt: Runtime = Depends(get_runtime)):
    return {
        "object": "list",
        "data": [{
            "id": _model_id(rt),
            "object": "model",
            "created": int(time.time()),
            "owned_by": "moss",
        }],
    }


@router.post("/chat/completions")
async def chat_completions(request: Request, rt: Runtime = Depends(get_runtime)):
    body = await request.json()
    stream = bool(body.get("stream", False))

    if not stream:
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        model = body.get("model") or _model_id(rt)
        vlm = rt.vlm
        messages = _translate_messages(body.get("messages") or [])
        req = ChatRequest(messages=messages, params=_translate_params(body), use_template=True)
        parts: List[str] = []
        async for delta in vlm.generate_stream(req):
            parts.append(delta)
        content = "".join(parts)
        return {
            "id": chat_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": _estimate_prompt_tokens(vlm, body.get("messages") or []),
                "completion_tokens": _estimate_completion_tokens(vlm, content),
                "total_tokens": 0,
            },
        }

    async def _event_gen():
        try:
            async for line in _chat_completions_sse(rt, body):
                yield line
        except Exception as exc:  # noqa: BLE001
            log.exception("openai chat failed: %s", exc)
            yield _sse({"error": {"message": str(exc), "type": "server_error"}})

    return StreamingResponse(
        _event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
