# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.watermarking import WatermarkConfig
from vllm.v1.watermarking.gumbel import (
    DualKeyGumbelWatermarker,
    GumbelWatermarker,
)
from vllm.v1.watermarking.synthid import SynthIDWatermarker
from vllm.v1.watermarking.watermarker import Watermarker


def create_watermarker(config: WatermarkConfig) -> Watermarker:
    if config.algorithm == "gumbel":
        return GumbelWatermarker(config.key, config.context_width, config.prf)
    if config.algorithm == "dual_key_gumbel":
        return DualKeyGumbelWatermarker(
            config.key,
            config.context_width,
            config.prf,
            config.alpha,
        )
    if config.algorithm == "synthid":
        return SynthIDWatermarker(
            key=config.key,
            context_width=config.context_width,
            depth=config.synthid_depth,
            prf=config.prf,
            strength=config.synthid_strength,
            max_probability=config.synthid_max_probability,
        )
    raise ValueError(f"Unknown watermarking algorithm: {config.algorithm}")
