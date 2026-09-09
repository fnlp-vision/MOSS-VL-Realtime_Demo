"""Sentence segmenter for token-streamed TTS input.

Ported from board voice_runtime.py TtsSession segmenter. Accumulates streamed LLM
text and cuts it into TTS-sized segments at sentence boundaries (>= min chars), or
force-cuts on soft/hard length limits at a nearby punctuation/space. This is what
lets the first sentence's audio start playing while later text is still arriving.
"""
from __future__ import annotations

import re
from typing import List

SENTENCE_END_RE = re.compile(r"([。！？!?；;\n]|(?<!\d)\.(?!\d))")

_BOUNDARY_CHARS = (" ", "\n", ",", "，", "、", ":", "：", ";", "；")


class Segmenter:
    def __init__(self, min_chars: int = 32, soft_chars: int = 72, max_chars: int = 160):
        self.min_chars = int(min_chars)
        self.soft_chars = int(soft_chars)
        self.max_chars = int(max_chars)
        self.buffer = ""

    def reset(self) -> None:
        self.buffer = ""

    def feed(self, text: str) -> List[str]:
        """Append text, return any newly-complete segments."""
        if not text:
            return []
        self.buffer += text
        return self._flush_ready(final=False)

    def flush(self) -> List[str]:
        """Return everything remaining as a final segment."""
        return self._flush_ready(final=True)

    def _flush_ready(self, final: bool) -> List[str]:
        out: List[str] = []
        while self.buffer:
            cut = self._find_cut(final=final)
            if cut <= 0:
                break
            text = self.buffer[:cut].strip()
            self.buffer = self.buffer[cut:].lstrip()
            if text:
                out.append(text)
        return out

    def _find_cut(self, final: bool) -> int:
        if final:
            # flush() must still honor max_chars: an over-long tail (a model
            # round that never hit a sentence boundary) would otherwise be
            # handed to TTS as one giant segment, which the engine's frame
            # cap hard-truncates — an audible mid-word cut. Cut at sentence
            # marks / boundaries first; only a truly unpunctuated remainder
            # shorter than max_chars goes out whole.
            if len(self.buffer) <= self.max_chars:
                return len(self.buffer)
            match = None
            for match in SENTENCE_END_RE.finditer(self.buffer):
                if match.end() >= self.max_chars:
                    break
            if match and match.end() >= self.min_chars:
                return match.end()
            return self._boundary_cut(self.max_chars)
        if len(self.buffer) < self.min_chars:
            return 0
        match = None
        for match in SENTENCE_END_RE.finditer(self.buffer):
            pass
        if match and match.end() >= self.min_chars:
            return match.end()
        if len(self.buffer) >= self.max_chars:
            return self._boundary_cut(self.max_chars)
        if len(self.buffer) >= self.soft_chars:
            return self._boundary_cut(self.soft_chars)
        return 0

    def _boundary_cut(self, limit: int) -> int:
        limit = min(max(self.min_chars, int(limit)), len(self.buffer))
        cut = -1
        for char in _BOUNDARY_CHARS:
            cut = max(cut, self.buffer.rfind(char, self.min_chars, limit + 1))
        if cut >= self.min_chars:
            return cut + 1
        return limit
