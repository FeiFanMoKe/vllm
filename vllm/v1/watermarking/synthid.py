# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SynthID-Text watermark generation and detection primitives.

Implements the non-distortionary variant of SynthID-Text
(https://www.nature.com/articles/s41586-024-08025-4): each of `depth`
layers casts a binary g-value vote on every candidate token, and the
sampling distribution is updated per layer as ``probs *= (1 + g - E[g])``
so the watermarked distribution matches the original one in expectation
over the secret key.

As in the reference implementation, all layer bits come from one keyed
pseudorandom evaluation per (context, token), and the update covers exactly
the tokens with finite logits.
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

_EXACT_BINOMIAL_MAX_N = 100_000
_MAX_DEPTH = 128
_FUSED_UPDATE_MAX_CANDIDATES = 4096


class SynthIDWatermarker(Watermarker):
    """Non-distortionary SynthID-Text tournament watermarker.

    Contexts with padding (negative token ids) belong to the first
    `context_width` generated tokens, whose context is incomplete. They are
    sampled without the watermark, as in the reference implementation.
    """

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        depth: int = 30,
        prf: WatermarkPRFName = "philox",
    ) -> None:
        if context_width < 1:
            raise ValueError("context_width must be positive")
        if not 1 <= depth <= _MAX_DEPTH:
            raise ValueError(f"depth must be between 1 and {_MAX_DEPTH}")
        self.prf = create_prf(prf, derive_watermark_key(key, b"synthid"))
        self.depth = depth
        self._context_width = context_width
        self._num_words = (depth + 31) // 32

    @property
    def context_width(self) -> int:
        return self._context_width

    def _transform_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the per-layer expected-g-value update to the logits.

        Only tokens with finite logits are candidates: the update is exact
        over the post top-k/top-p support, and every other token stays at
        ``-inf``. One PRF evaluation serves all layers, so the support size
        controls the per-step cost.
        """
        logits = logits.to(torch.float32)
        finite_counts = torch.isfinite(logits).sum(dim=-1)
        num_candidates = int(finite_counts.max())
        if num_candidates == 0:
            return logits

        if num_candidates == logits.shape[-1]:
            candidates = logits
            indices = None
            tokens = torch.arange(logits.shape[-1], device=logits.device)
        else:
            candidates, indices = logits.topk(num_candidates, dim=-1)
            tokens = indices

        logp = candidates.log_softmax(dim=-1)
        words = self.prf.bits(contexts, tokens, self._num_words)
        logp = self._apply_expected_g_update(logp, words)
        logp = torch.where(
            torch.isfinite(logp),
            logp,
            torch.full_like(logp, float("-inf")),
        )

        if indices is None:
            out = logp
        else:
            out = torch.full_like(logits, float("-inf"))
            out = out.scatter(-1, indices, logp)
        return out

    def _apply_expected_g_update(
        self, logp: torch.Tensor, words: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        if (
            logp.device.type == "cuda"
            and self._num_words == 1
            and logp.shape[1] <= _FUSED_UPDATE_MAX_CANDIDATES
        ):
            from vllm.v1.worker.gpu.sample.watermark import (
                HAS_TRITON,
                synthid_update,
            )

            if HAS_TRITON:
                return synthid_update(logp, words[0], self.depth)
        for layer in range(self.depth):
            g = ((words[layer // 32] >> (layer % 32)) & 1).to(logp.dtype)
            g_mass = (g * logp.exp()).sum(dim=-1, keepdim=True)
            logp = logp + torch.log1p(g - g_mass)
        return logp

    def _mixed_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        skip_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        transformed = self._transform_logits(logits, contexts)
        skip = (contexts < 0).any(dim=-1)
        if skip_mask is not None:
            skip = skip | skip_mask
        return torch.where(skip.unsqueeze(-1), logits, transformed)

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler | None = None,
        skip_mask: torch.Tensor | None = None,
    ) -> WatermarkSample:
        if random_sampler is None:
            return self._sample_watermarked(logits, contexts)
        transformed = self._mixed_logits(logits, contexts, skip_mask)
        return WatermarkSample(random_sampler(transformed), transformed)

    def _sample_watermarked(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> WatermarkSample:
        transformed = self._mixed_logits(logits, contexts, None)
        token_ids = torch.multinomial(
            torch.softmax(transformed, dim=-1), num_samples=1
        ).squeeze(-1)
        return WatermarkSample(token_ids, transformed)


class SynthIDWatermarkDetector(WatermarkDetector):
    """G-value count detector with an analytic binomial p-value.

    Under the null hypothesis each scored token receives independent fair
    coin flips across all layers, so the total g-value count follows a
    Binomial(num_scored_tokens * depth, 0.5) distribution. The first
    `context_width` tokens are not scored because their context is
    incomplete; pass only the generated tokens, not the prompt.
    """

    skip_partial_contexts = True

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        depth: int = 30,
        p_value_threshold: float = 0.01,
        prf: WatermarkPRFName = "philox",
        deduplicate_contexts: bool = True,
    ) -> None:
        if not 1 <= depth <= _MAX_DEPTH:
            raise ValueError(f"depth must be between 1 and {_MAX_DEPTH}")
        super().__init__(context_width, p_value_threshold, deduplicate_contexts)
        self.prf: WatermarkPRF = create_prf(prf, derive_watermark_key(key, b"synthid"))
        self.depth = depth
        self._num_words = (depth + 31) // 32

    def _score_tokens(
        self, contexts: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        words = self.prf.bits(contexts, targets.unsqueeze(-1), self._num_words)
        score = torch.zeros(contexts.shape[0], dtype=torch.float64)
        for layer in range(self.depth):
            bits = ((words[layer // 32] >> (layer % 32)) & 1).squeeze(-1)
            score += bits.to(torch.float64)
        return score

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
    if k > n:
        return 0.0
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
