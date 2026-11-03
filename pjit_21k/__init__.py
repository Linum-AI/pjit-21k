# Copyright 2026 Linum Inc. Licensed under the Apache License, Version 2.0.

"""
File: __init__.py
Description: P-JiT 21K — a pixel-space text-to-image diffusion transformer by Linum, with a
    prompt upsampler.
"""

from pjit_21k.attention import active_attention_backend, set_attention_backend
from pjit_21k.config import ASPECT_RATIOS, PyramidJiTConfig, SamplerConfig
from pjit_21k.model import PyramidJiT
from pjit_21k.noise import randn_as_on_sm_count
from pjit_21k.sampler import generate
from pjit_21k.text_encoder import QwenTextEncoder
from pjit_21k.upsampler import UpsampleResult, upsample_prompt

__all__ = [
    "ASPECT_RATIOS",
    "PyramidJiT",
    "PyramidJiTConfig",
    "QwenTextEncoder",
    "SamplerConfig",
    "UpsampleResult",
    "active_attention_backend",
    "generate",
    "randn_as_on_sm_count",
    "set_attention_backend",
    "upsample_prompt",
]
