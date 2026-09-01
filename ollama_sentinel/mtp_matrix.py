"""MTP / speculative-decoding platform notes (research-backed, not guarantees)."""

from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class MtpPlatformNote:
    platform: str
    acceleration_likely: bool
    summary: str


def mtp_platform_note(*, darwin: bool | None = None) -> MtpPlatformNote:
    """Return fleet-measured expectations for MTP on Ollama."""
    if darwin is None:
        darwin = sys.platform == "darwin"
    if darwin:
        return MtpPlatformNote(
            platform="darwin",
            acceleration_likely=True,
            summary=(
                "MLX path may use embedded MTP on supported families (e.g. Gemma 4). "
                "Qwen MTP tensors may still be ignored depending on Ollama build."
            ),
        )
    if sys.platform == "win32":
        return MtpPlatformNote(
            platform="win32",
            acceleration_likely=False,
            summary=(
                "Windows CUDA GGUF path often ignores Qwen MTP tensors even when "
                "draft_num_predict is set. Verify with your Ollama version; llama.cpp "
                "server is the fallback for guaranteed MTP."
            ),
        )
    return MtpPlatformNote(
        platform=sys.platform,
        acceleration_likely=False,
        summary=(
            "Linux ROCm/CUDA MTP support varies by Ollama build. Architectural MTP "
            "tensors plus draft_num_predict are necessary but not sufficient for speedup."
        ),
    )
