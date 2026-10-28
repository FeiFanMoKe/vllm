# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config.watermarking import WatermarkConfig
from vllm.v1.watermarking import create_watermarker
from vllm.v1.watermarking.synthid import (
    SynthIDWatermarkDetector,
    SynthIDWatermarker,
    _binomial_survival,
)

KEY = 42
VOCAB_SIZE = 512
CONTEXT_WIDTH = 4
DEPTH = 30


def _g_bits(
    prf, contexts: torch.Tensor, tokens: torch.Tensor, depth: int
) -> torch.Tensor:
    words = prf.bits(contexts, tokens, (depth + 31) // 32)
    return torch.stack(
        [((words[layer // 32] >> (layer % 32)) & 1) for layer in range(depth)]
    )


def _reference_update(
    logits: torch.Tensor,
    contexts: torch.Tensor,
    watermarker: SynthIDWatermarker,
) -> torch.Tensor:
    """Probability-domain reference, computed one layer at a time.

    Mirrors update_scores() in DeepMind's synthid-text repository; used to
    cross-check the vectorized log-domain implementation.
    """
    probs = torch.softmax(logits.to(torch.float32), dim=-1)
    vocabulary = torch.arange(logits.shape[-1], device=logits.device)
    bits = _g_bits(watermarker.prf, contexts, vocabulary, watermarker.depth)
    for layer in range(watermarker.depth):
        g = bits[layer].to(probs.dtype)
        for row in range(probs.shape[0]):
            g_mass = (g[row] * probs[row]).sum()
            probs[row] = probs[row] * (1.0 + g[row] - g_mass)
    log_probs = torch.log(probs)
    return torch.where(
        torch.isfinite(logits),
        log_probs,
        torch.full_like(log_probs, float("-inf")),
    )


def _random_logits(batch: int, vocab: int, seed: int) -> torch.Tensor:
    return torch.randn(batch, vocab, generator=torch.Generator().manual_seed(seed))


def _random_contexts(batch: int, seed: int) -> torch.Tensor:
    return torch.randint(
        0,
        VOCAB_SIZE,
        (batch, CONTEXT_WIDTH),
        generator=torch.Generator().manual_seed(seed),
    )


def _assert_same_distribution(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    assert torch.allclose(
        torch.softmax(actual, dim=-1),
        torch.softmax(expected, dim=-1),
        atol=1e-7,
        rtol=1e-4,
    )


def test_transform_matches_reference_implementation():
    watermarker = SynthIDWatermarker(KEY, CONTEXT_WIDTH, DEPTH)
    logits = _random_logits(3, VOCAB_SIZE, seed=0)
    contexts = _random_contexts(3, seed=0)
    expected = _reference_update(logits, contexts, watermarker)
    actual = watermarker._transform_logits(logits, contexts)
    _assert_same_distribution(actual, expected)


def test_transform_preserves_truncated_support():
    """Tokens excluded by top-k/top-p must stay at -inf after watermarking."""
    watermarker = SynthIDWatermarker(KEY, CONTEXT_WIDTH, DEPTH)
    logits = _random_logits(2, VOCAB_SIZE, seed=1)
    logits[:, VOCAB_SIZE // 2 :] = float("-inf")
    contexts = _random_contexts(2, seed=1)
    transformed = watermarker._transform_logits(logits, contexts)
    assert torch.isneginf(transformed[:, VOCAB_SIZE // 2 :]).all()
    assert torch.isfinite(transformed[:, : VOCAB_SIZE // 2]).all()


def test_transform_exact_with_per_row_support():
    """Rows with fewer finite tokens than the batch maximum stay exact."""
    watermarker = SynthIDWatermarker(KEY, CONTEXT_WIDTH, DEPTH)
    logits = _random_logits(3, VOCAB_SIZE, seed=2)
    logits[0, 400:] = float("-inf")
    logits[1, 250:] = float("-inf")
    logits[2, 10:] = float("-inf")
    contexts = _random_contexts(3, seed=2)
    expected = _reference_update(logits, contexts, watermarker)
    actual = watermarker._transform_logits(logits, contexts)
    _assert_same_distribution(actual, expected)


def test_transform_all_masked_row_has_no_nan():
    watermarker = SynthIDWatermarker(KEY, CONTEXT_WIDTH, DEPTH)
    logits = _random_logits(2, VOCAB_SIZE, seed=3)
    logits[0] = float("-inf")
    contexts = _random_contexts(2, seed=3)
    transformed = watermarker._transform_logits(logits, contexts)
    assert not torch.isnan(transformed).any()
    assert torch.isneginf(transformed[0]).all()
    assert torch.isfinite(transformed[1]).all()


def test_g_bits_deterministic_and_balanced_across_layers():
    watermarker = SynthIDWatermarker(KEY, CONTEXT_WIDTH, DEPTH)
    contexts = _random_contexts(64, seed=4)
    vocabulary = torch.arange(VOCAB_SIZE)
    bits = _g_bits(watermarker.prf, contexts, vocabulary, watermarker.depth)
    assert torch.equal(
        bits, _g_bits(watermarker.prf, contexts, vocabulary, watermarker.depth)
    )
    fractions = bits.to(torch.float64).mean(dim=(1, 2))
    assert (fractions - 0.5).abs().max().item() < 0.02


def test_g_bits_across_layers_are_uncorrelated():
    watermarker = SynthIDWatermarker(KEY, CONTEXT_WIDTH, DEPTH)
    contexts = _random_contexts(256, seed=5)
    vocabulary = torch.arange(VOCAB_SIZE)
    g = _g_bits(watermarker.prf, contexts, vocabulary, watermarker.depth).double()
    means = g.mean(dim=(1, 2), keepdim=True)
    centered = g - means
    covariance = torch.einsum("lbc,kbc->lk", centered, centered)
    covariance /= g.shape[1] * g.shape[2]
    variance = covariance.diagonal()
    correlation = covariance / torch.sqrt(torch.outer(variance, variance))
    off_diagonal = correlation[~torch.eye(DEPTH, dtype=torch.bool)]
    assert off_diagonal.abs().max().item() < 0.05


def test_watermarked_distribution_unbiased_in_expectation():
    """Non-distortion: averaging over independent keys recovers the original.

    Tested with a single layer: the per-layer update is where an
    implementation error would live, and composition across independent
    mean-1 layers is unbiased by construction. With depth=30 the per-draw
    variance compounds multiplicatively (~1.25**30), so a sample average
    cannot statistically distinguish bias from noise at test scale.
    """
    watermarker = SynthIDWatermarker(KEY, CONTEXT_WIDTH, depth=1)
    logits = _random_logits(1, 128, seed=7)
    base_probs = torch.softmax(logits, dim=-1).squeeze(0)
    num_draws = 5000
    contexts = torch.randint(
        0,
        VOCAB_SIZE,
        (num_draws, CONTEXT_WIDTH),
        generator=torch.Generator().manual_seed(7),
    )
    transformed = watermarker._transform_logits(logits.expand(num_draws, -1), contexts)
    mean_probs = torch.softmax(transformed, dim=-1).mean(dim=0)
    significant = base_probs > 0.01
    relative_error = ((mean_probs - base_probs).abs() / base_probs)[significant]
    assert relative_error.max().item() < 0.05


def test_watermarked_generation_is_detectable():
    torch.manual_seed(0)
    watermarker = SynthIDWatermarker(KEY, CONTEXT_WIDTH, DEPTH)
    detector = SynthIDWatermarkDetector(KEY, CONTEXT_WIDTH, DEPTH)
    token_ids: list[int] = []
    for step in range(400):
        logits = _random_logits(1, VOCAB_SIZE, seed=step)
        prefix = ([-1] * CONTEXT_WIDTH + token_ids)[-CONTEXT_WIDTH:]
        contexts = torch.tensor([prefix])
        token_ids.append(watermarker.sample(logits, contexts).token_ids.item())
    result = detector.detect(token_ids)
    assert result.is_watermarked
    assert result.p_value < 0.01


def test_unwatermarked_generation_is_not_flagged():
    generator = torch.Generator().manual_seed(0)
    token_ids = torch.randint(0, VOCAB_SIZE, (400,), generator=generator).tolist()
    detector = SynthIDWatermarkDetector(KEY, CONTEXT_WIDTH, DEPTH)
    result = detector.detect(token_ids)
    assert not result.is_watermarked
    assert result.p_value > 0.01


def test_wrong_key_does_not_detect():
    torch.manual_seed(0)
    watermarker = SynthIDWatermarker(KEY, CONTEXT_WIDTH, DEPTH)
    token_ids: list[int] = []
    for step in range(400):
        logits = _random_logits(1, VOCAB_SIZE, seed=step)
        prefix = ([-1] * CONTEXT_WIDTH + token_ids)[-CONTEXT_WIDTH:]
        contexts = torch.tensor([prefix])
        token_ids.append(watermarker.sample(logits, contexts).token_ids.item())
    detector = SynthIDWatermarkDetector(KEY + 1, CONTEXT_WIDTH, DEPTH)
    assert not detector.detect(token_ids).is_watermarked


def test_detector_skips_partial_contexts():
    detector = SynthIDWatermarkDetector(KEY, CONTEXT_WIDTH, DEPTH)
    result = detector.detect(list(range(10)))
    assert result.num_scored_tokens == 10 - CONTEXT_WIDTH
    assert detector.detect([1, 2, 3]).num_scored_tokens == 0


def test_detector_handles_empty_input():
    detector = SynthIDWatermarkDetector(KEY, CONTEXT_WIDTH, DEPTH)
    result = detector.detect([])
    assert result.num_scored_tokens == 0
    assert not result.is_watermarked


def test_sample_skips_rows_with_partial_contexts():
    watermarker = SynthIDWatermarker(KEY, CONTEXT_WIDTH, DEPTH)
    logits = _random_logits(2, 128, seed=8)

    sampler_calls = 0

    def argmax_sampler(logits: torch.Tensor) -> torch.Tensor:
        nonlocal sampler_calls
        sampler_calls += 1
        return logits.argmax(dim=-1)

    contexts = torch.tensor([[-1, -1, -1, -1], [0, 1, 2, 3]])
    sampled = watermarker.sample(logits, contexts, argmax_sampler)
    assert sampler_calls == 1
    assert sampled.token_ids[0] == logits[0].argmax()
    assert torch.equal(sampled.logits[0], logits[0])

    skipped = watermarker.sample(
        logits, contexts, argmax_sampler, skip_mask=torch.tensor([False, True])
    )
    assert torch.equal(skipped.logits[1], logits[1])


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        SynthIDWatermarker(KEY, context_width=0)
    with pytest.raises(ValueError):
        SynthIDWatermarker(KEY, depth=0)
    with pytest.raises(ValueError):
        SynthIDWatermarker(KEY, depth=129)
    with pytest.raises(ValueError):
        SynthIDWatermarkDetector(KEY, depth=0)
    with pytest.raises(ValueError):
        SynthIDWatermarkDetector(KEY, depth=129)


def test_binomial_survival_matches_closed_form():
    # P(X >= k) for X ~ Binomial(4, 0.5) has an exact closed form.
    assert _binomial_survival(4, 4) == pytest.approx(1 / 16)
    assert _binomial_survival(3, 4) == pytest.approx(5 / 16)
    assert _binomial_survival(0, 4) == 1.0
    assert _binomial_survival(5, 4) == 0.0
    assert _binomial_survival(6, 4) == 0.0


def test_factory_creates_synthid_watermarker():
    config = WatermarkConfig(key=KEY, algorithm="synthid", synthid_depth=8)
    watermarker = create_watermarker(config)
    assert isinstance(watermarker, SynthIDWatermarker)
    assert watermarker.depth == 8
