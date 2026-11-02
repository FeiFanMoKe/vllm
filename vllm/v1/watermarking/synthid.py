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
from collections import deque
from dataclasses import replace

import numpy as np
import torch

from vllm.config.watermarking import (
    WatermarkConfig,
    WatermarkPRFName,
    derive_watermark_key,
)
from vllm.v1.watermarking.detector import WatermarkDetection, WatermarkDetector
from vllm.v1.watermarking.prfs import WatermarkPRF, create_prf
from vllm.v1.watermarking.validation import validate_context_width
from vllm.v1.watermarking.watermarker import (
    RandomSampler,
    Watermarker,
    WatermarkSample,
)

_EXACT_BINOMIAL_MAX_N = 100_000
_MAX_DEPTH = 128
_FUSED_UPDATE_MAX_CANDIDATES = 4096
_FULL_SUPPORT_ROW_CHUNK = 8


class SynthIDWatermarker(Watermarker):
    """Non-distortionary SynthID-Text tournament watermarker.

    Contexts with padding (negative token ids) belong to the first
    `context_width` generated tokens, whose context is incomplete. They are
    sampled without the watermark, as in the reference implementation.
    """

    supports_fused_sampling = True

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        depth: int = 30,
        prf: WatermarkPRFName = "philox",
        strength: float = 1.0,
        max_probability: float = 1.0,
    ) -> None:
        validate_context_width(context_width)
        if not 1 <= depth <= _MAX_DEPTH:
            raise ValueError(f"depth must be between 1 and {_MAX_DEPTH}")
        if not 0 < strength <= 1:
            raise ValueError("strength must be in (0, 1]")
        if not 0 < max_probability <= 1:
            raise ValueError("max_probability must be in (0, 1]")
        self.prf = create_prf(prf, derive_watermark_key(key, b"synthid"))
        if not self.prf.supports_bits:
            raise ValueError(
                f"{type(self.prf).__name__} does not provide packed g-value bits"
            )
        self.depth = depth
        self.strength = strength
        self.max_probability = max_probability
        self._context_width = context_width
        self._num_words = (depth + 31) // 32

    @property
    def context_width(self) -> int:
        return self._context_width

    def _transform_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        candidate_bound: int | None = None,
    ) -> torch.Tensor:
        """Apply the per-layer expected-g-value update to the logits.

        Only tokens with finite logits are candidates: the update is exact
        over the post top-k/top-p support, and every other token stays at
        ``-inf``. One PRF evaluation serves all layers, so the support size
        controls the per-step cost. ``candidate_bound`` is an upper bound on
        that support; without it, the exact size requires a device
        synchronization. The bound assumes top-k kept at most ``top_k`` tokens,
        which degenerate rows with all-equal finite logits violate (see the
        no-mask fallback in ``topk_topp_triton``).
        """
        logits = logits.to(torch.float32)
        if candidate_bound is None:
            finite_counts = torch.isfinite(logits).sum(dim=-1)
            num_candidates = int(finite_counts.max())
        else:
            if candidate_bound < 1:
                raise ValueError("candidate_bound must be positive")
            num_candidates = min(candidate_bound, logits.shape[-1])
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

    def _transform_batch(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        support_bounds: np.ndarray,
    ) -> torch.Tensor:
        """Apply the update row-group by row-group.

        ``support_bounds`` holds one entry per row: ``-1`` leaves the row
        unchanged, ``0`` uses the exact support count, and a positive value is
        an upper bound.
        """
        values = np.unique(support_bounds)
        if values.size == 1 and 0 < values[0] < logits.shape[-1]:
            return self._transform_logits(logits, contexts, int(values[0]))

        out = logits.to(torch.float32).clone()
        for value in values:
            if value < 0:
                continue
            rows = np.flatnonzero(support_bounds == value)
            index = torch.from_numpy(rows).to(device=logits.device, dtype=torch.int64)
            bound = None if value == 0 else int(value)
            full_support = bound is not None and bound >= logits.shape[-1]
            chunk_size = _FULL_SUPPORT_ROW_CHUNK if full_support else len(rows)
            for start in range(0, len(rows), chunk_size):
                chunk = index[start : start + chunk_size]
                out[chunk] = self._transform_logits(
                    logits[chunk], contexts[chunk], bound
                )
        return out

    def _apply_expected_g_update(
        self, logp: torch.Tensor, words: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        # TODO: the eager and Triton paths round g_mass differently at the
        # clamp boundary, so near-degenerate rows can differ across devices.
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
                return synthid_update(logp, words[0], self.depth, self.strength)
        for layer in range(self.depth):
            g = ((words[layer // 32] >> (layer % 32)) & 1).to(logp.dtype)
            g_mass = (g * logp.exp()).sum(dim=-1, keepdim=True)
            bias = self.strength * (g - g_mass)
            logp = logp + torch.log1p(bias.clamp_min(-1.0))
        return logp

    def _mixed_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        skip_mask: torch.Tensor | None,
        support_bounds: np.ndarray | None = None,
    ) -> torch.Tensor:
        if support_bounds is None:
            transformed = self._transform_logits(logits, contexts)
        else:
            transformed = self._transform_batch(logits, contexts, support_bounds)
        skip = (contexts < 0).any(dim=-1)
        if skip_mask is not None:
            skip = skip | skip_mask
        if self.max_probability < 1.0:
            # TODO: reuse the pre-update log-probs from the transform instead
            # of a second full-vocabulary softmax. Skipping the transform for
            # gated rows needs row compaction (which synchronizes) or a
            # device-side kernel.
            max_prob = logits.to(torch.float32).softmax(dim=-1).amax(dim=-1)
            skip = skip | (max_prob > self.max_probability)
        return torch.where(skip.unsqueeze(-1), logits, transformed)

    def transform_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        skip_mask: torch.Tensor | None = None,
        support_bounds: np.ndarray | None = None,
    ) -> torch.Tensor:
        return self._mixed_logits(logits, contexts, skip_mask, support_bounds)

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler | None = None,
        skip_mask: torch.Tensor | None = None,
        support_bounds: np.ndarray | None = None,
    ) -> WatermarkSample:
        """Sample from the SynthID-adjusted distribution.

        The returned logits are the input processed logits, as for Gumbel-max,
        so processed-logit/logprob modes do not expose the key-dependent
        adjustment.
        """
        if random_sampler is None:
            return self._sample_watermarked(logits, contexts, skip_mask, support_bounds)
        transformed = self._mixed_logits(logits, contexts, skip_mask, support_bounds)
        return WatermarkSample(random_sampler(transformed), logits)

    def _sample_watermarked(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        skip_mask: torch.Tensor | None = None,
        support_bounds: np.ndarray | None = None,
    ) -> WatermarkSample:
        transformed = self._mixed_logits(logits, contexts, skip_mask, support_bounds)
        token_ids = torch.multinomial(
            torch.softmax(transformed, dim=-1), num_samples=1
        ).squeeze(-1)
        return WatermarkSample(token_ids, logits)


class SynthIDWatermarkDetector(WatermarkDetector):
    """G-value count detector with an analytic binomial p-value.

    Under the null hypothesis each scored token receives independent fair
    coin flips across all layers, so the total g-value count follows a
    Binomial(num_scored_tokens * depth, 0.5) distribution. The first
    `context_width` tokens are not scored because their context is
    incomplete; pass only the generated tokens, not the prompt.

    ``detect`` keeps per-call statistics on the instance, so one detector must
    not serve concurrent calls. TODO: return the statistics from
    ``_prepare_inputs`` instead, and window ``seen_pairs`` for long documents.
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
        deduplicate_contexts_max_history: int | None = 1024,
        eos_token_id: int | None = None,
    ) -> None:
        validate_context_width(context_width)
        if not 1 <= depth <= _MAX_DEPTH:
            raise ValueError(f"depth must be between 1 and {_MAX_DEPTH}")
        if (
            deduplicate_contexts_max_history is not None
            and deduplicate_contexts_max_history < 1
        ):
            raise ValueError(
                "deduplicate_contexts_max_history must be positive or None"
            )
        super().__init__(context_width, p_value_threshold, deduplicate_contexts)
        self.prf: WatermarkPRF = create_prf(prf, derive_watermark_key(key, b"synthid"))
        if not self.prf.supports_bits:
            raise ValueError(
                f"{type(self.prf).__name__} does not provide packed g-value bits"
            )
        self.depth = depth
        self._num_words = (depth + 31) // 32
        self.deduplicate_contexts_max_history = deduplicate_contexts_max_history
        self.eos_token_id = eos_token_id
        self._repeated_contexts = 0
        self._duplicate_context_tokens = 0

    @classmethod
    def from_config(
        cls,
        config: WatermarkConfig,
        *,
        p_value_threshold: float = 0.01,
        deduplicate_contexts: bool = True,
        eos_token_id: int | None = None,
    ) -> "SynthIDWatermarkDetector":
        """Build a detector whose SynthID parameters match ``config``.

        ``deduplicate_contexts`` controls scoring independence and is unrelated
        to the generation-side ``config.deduplicate_contexts`` scope.
        """
        return cls(
            key=config.key,
            context_width=config.context_width,
            depth=config.synthid_depth,
            p_value_threshold=p_value_threshold,
            prf=config.prf,
            deduplicate_contexts=deduplicate_contexts,
            deduplicate_contexts_max_history=config.deduplicate_contexts_max_history,
            eos_token_id=eos_token_id,
        )

    def _prepare_inputs(
        self, token_ids: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        contexts: list[tuple[int, ...]] = []
        targets = []
        prefix = [-1] * self.context_width
        history: deque[tuple[int, ...]] = deque()
        history_counts: dict[tuple[int, ...], int] = {}
        seen_pairs: set[tuple[tuple[int, ...], int]] = set()
        repeated = 0
        duplicates = 0
        for token_id in token_ids:
            if self.eos_token_id is not None and token_id == self.eos_token_id:
                break
            context = tuple(prefix[-self.context_width :])
            if self.skip_partial_contexts and any(token < 0 for token in context):
                prefix.append(token_id)
                continue
            if self.deduplicate_contexts:
                is_repeated = context in history_counts
                if (
                    self.deduplicate_contexts_max_history is not None
                    and len(history) == self.deduplicate_contexts_max_history
                ):
                    evicted = history.popleft()
                    remaining = history_counts[evicted] - 1
                    if remaining:
                        history_counts[evicted] = remaining
                    else:
                        del history_counts[evicted]
                history.append(context)
                history_counts[context] = history_counts.get(context, 0) + 1
                if is_repeated:
                    repeated += 1
                    prefix.append(token_id)
                    continue
                pair = (context, token_id)
                if pair in seen_pairs:
                    duplicates += 1
                    prefix.append(token_id)
                    continue
                seen_pairs.add(pair)
            contexts.append(context)
            targets.append(token_id)
            prefix.append(token_id)

        self._repeated_contexts = repeated
        self._duplicate_context_tokens = duplicates
        if not contexts:
            return (
                torch.empty((0, self.context_width), dtype=torch.int64),
                torch.empty(0, dtype=torch.int64),
            )
        return (
            torch.tensor(contexts, dtype=torch.int64).reshape(-1, self.context_width),
            torch.tensor(targets, dtype=torch.int64),
        )

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

    def _get_statistics(
        self, score: float, num_scored_tokens: int
    ) -> tuple[float, float | None]:
        trials = num_scored_tokens * self.depth
        if trials == 0:
            return 1.0, 0.0
        k = min(int(score), trials)
        if trials <= _EXACT_BINOMIAL_MAX_N:
            log_p = _binomial_log_survival(k, trials)
        else:
            z = (score - 0.5 - trials / 2) / math.sqrt(trials / 4)
            log_p = _normal_log_survival(z)
        return math.exp(log_p), log_p

    def _get_p_value(self, score: float, num_scored_tokens: int) -> float:
        return self._get_statistics(score, num_scored_tokens)[0]

    def detect(self, token_ids: list[int]) -> WatermarkDetection:
        result = super().detect(token_ids)
        trials = result.num_scored_tokens * self.depth
        z = (
            0.0
            if trials == 0
            else (result.score - trials / 2) / math.sqrt(trials / 4)
        )
        return replace(
            result,
            log_p_value=0.0 if trials == 0 else result.log_p_value,
            z_score=z,
            repeated_contexts=self._repeated_contexts,
            duplicate_context_tokens=self._duplicate_context_tokens,
        )


def _binomial_survival(k: int, n: int) -> float:
    """Return P(X >= k) for X ~ Binomial(n, 0.5)."""
    return math.exp(_binomial_log_survival(k, n))


def _binomial_log_survival(count: int, trials: int) -> float:
    """Return log P(X >= count) for X ~ Binomial(trials, 0.5)."""
    if count <= 0:
        return 0.0
    if count > trials:
        return -math.inf
    log_2_pow_n = trials * math.log(2)

    def log_pmf(i: int) -> float:
        return (
            math.lgamma(trials + 1)
            - math.lgamma(i + 1)
            - math.lgamma(trials - i + 1)
            - log_2_pow_n
        )

    log_terms = [log_pmf(count)]
    for i in range(count + 1, trials + 1):
        log_terms.append(log_terms[-1] + math.log(trials - i + 1) - math.log(i))
    peak = max(log_terms)
    return min(
        0.0,
        peak + math.log(math.fsum(math.exp(t - peak) for t in log_terms)),
    )


def _normal_log_survival(z: float) -> float:
    """Return log P(X >= z) for a standard normal X."""
    if z < 38:
        return math.log(0.5 * math.erfc(z / math.sqrt(2)))
    return -z * z / 2 - math.log(z * math.sqrt(2 * math.pi))
