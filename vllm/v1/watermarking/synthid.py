# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SynthID-Text watermark generation and detection primitives.

Implements the non-distortionary variant of SynthID-Text
(https://www.nature.com/articles/s41586-024-08025-4): each of `depth`
independently keyed layers casts a binary g-value vote on every candidate
token, and the sampling distribution is updated per layer as
``probs *= (1 + g - E[g])`` so the watermarked distribution matches the
original one in expectation over the secret key.
"""

import math

import torch

from vllm.config.watermarking import WatermarkPRFName, derive_watermark_key
from vllm.v1.watermarking.detector import WatermarkDetector
from vllm.v1.watermarking.prfs import WatermarkPRF, create_prf
from vllm.v1.watermarking.watermarker import (
    RandomSampler,
    Watermarker,
    WatermarkSample,
)

_NEG_INF_SUBSTITUTE = -1e12
_EXACT_BINOMIAL_MAX_N = 100_000


def _layer_prfs(key: int, depth: int, prf: WatermarkPRFName) -> list[WatermarkPRF]:
    return [
        create_prf(prf, derive_watermark_key(key, f"synthid_layer_{i}".encode()))
        for i in range(depth)
    ]


class SynthIDWatermarker(Watermarker):
    """Non-distortionary SynthID-Text tournament watermarker."""

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        depth: int = 30,
        prf: WatermarkPRFName = "philox",
    ) -> None:
        if context_width < 1:
            raise ValueError("context_width must be positive")
        if depth < 1:
            raise ValueError("depth must be positive")
        self.layer_prfs = _layer_prfs(key, depth, prf)
        self.depth = depth
        self._context_width = context_width

    @property
    def context_width(self) -> int:
        return self._context_width

    def _transform_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the per-layer expected-g-value update to the logits.

        g-values are computed only over tokens with nonzero probability
        (finite logits after top-k/top-p): excluded tokens contribute
        nothing to the update, so gathering the support first avoids one
        full-vocabulary PRF evaluation per layer. Falls back to the dense
        path when no truncation is in effect.
        """
        vocab_size = logits.shape[-1]
        # One small device-to-host read per call to size the support.
        max_support = int(torch.isfinite(logits).sum(dim=-1).max().item())
        if max_support == vocab_size:
            vocabulary = torch.arange(vocab_size, device=logits.device)
            return self._transform_dense(logits, contexts, vocabulary)

        indices = torch.topk(logits, max_support, dim=-1).indices
        candidate_logits = logits.gather(-1, indices)
        transformed = self._transform_dense(candidate_logits, contexts, indices)
        output = torch.full_like(logits.to(torch.float32), _NEG_INF_SUBSTITUTE)
        return output.scatter(-1, indices, transformed)

    def _transform_dense(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.softmax(logits.to(torch.float32), dim=-1)
        for layer_prf in self.layer_prfs:
            g = (layer_prf.uniform(contexts, token_ids) >= 0.5).to(probs.dtype)
            g_mass = (g * probs).sum(dim=-1, keepdim=True)
            probs = probs * (1.0 + g - g_mass)
        log_probs = torch.log(probs)
        return torch.where(
            torch.isfinite(log_probs),
            log_probs,
            torch.full_like(log_probs, _NEG_INF_SUBSTITUTE),
        )

    def _sample_watermarked(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> WatermarkSample:
        transformed = self._transform_logits(logits, contexts)
        token_ids = torch.multinomial(
            torch.softmax(transformed, dim=-1), num_samples=1
        ).squeeze(-1)
        return WatermarkSample(token_ids, transformed)

    def _try_sample_mixed(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        skip_mask: torch.Tensor,
        random_sampler: RandomSampler,
    ) -> WatermarkSample:
        transformed = self._transform_logits(logits, contexts)
        token_ids = torch.where(
            skip_mask, random_sampler(logits), random_sampler(transformed)
        )
        output_logits = torch.where(skip_mask.unsqueeze(-1), logits, transformed)
        return WatermarkSample(token_ids, output_logits)


class SynthIDWatermarkDetector(WatermarkDetector):
    """Mean g-value detector with an analytic binomial p-value.

    Under the null hypothesis each scored token receives independent fair
    coin flips across all layers, so the total g-value count follows a
    Binomial(num_scored_tokens * depth, 0.5) distribution.
    """

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        depth: int = 30,
        p_value_threshold: float = 0.01,
        prf: WatermarkPRFName = "philox",
        deduplicate_contexts: bool = True,
    ) -> None:
        if depth < 1:
            raise ValueError("depth must be positive")
        super().__init__(context_width, p_value_threshold, deduplicate_contexts)
        self.layer_prfs = _layer_prfs(key, depth, prf)
        self.depth = depth

    def _score_tokens(
        self, contexts: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        targets = targets.unsqueeze(-1)
        g_values = torch.stack(
            [
                (prf.uniform(contexts, targets).squeeze(-1) >= 0.5)
                for prf in self.layer_prfs
            ],
            dim=-1,
        )
        return g_values.to(torch.float64).sum(dim=-1)

    def _aggregate_scores(self, token_scores: torch.Tensor) -> float:
        return token_scores.sum().item()

    def _get_p_value(self, score: float, num_scored_tokens: int) -> float:
        n = num_scored_tokens * self.depth
        k = min(int(score), n)
        if n <= _EXACT_BINOMIAL_MAX_N:
            return _binomial_survival(k, n)
        z = (score - 0.5 - n / 2) / math.sqrt(n / 4)
        return 0.5 * math.erfc(z / math.sqrt(2))


def _binomial_survival(k: int, n: int) -> float:
    """Return P(X >= k) for X ~ Binomial(n, 0.5), computed in log space."""
    if k <= 0:
        return 1.0
    log_2_pow_n = n * math.log(2)

    def log_pmf(i: int) -> float:
        return (
            math.lgamma(n + 1)
            - math.lgamma(i + 1)
            - math.lgamma(n - i + 1)
            - log_2_pow_n
        )

    log_terms = [log_pmf(k)]
    for i in range(k + 1, n + 1):
        log_terms.append(log_terms[-1] + math.log(n - i + 1) - math.log(i))
    max_log = max(log_terms)
    return min(1.0, math.exp(max_log) * sum(math.exp(t - max_log) for t in log_terms))
