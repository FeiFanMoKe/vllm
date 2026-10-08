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


def _naive_expected_g_update(
    logits: torch.Tensor,
    contexts: torch.Tensor,
    watermarker: SynthIDWatermarker,
) -> torch.Tensor:
    """Reference expected-g-value update, computed one layer at a time.

    Mirrors update_scores() in DeepMind's synthid-text repository; used to
    cross-check the vectorized implementation.
    """
    vocabulary = torch.arange(logits.shape[-1], device=logits.device)
    probs = torch.softmax(logits.to(torch.float32), dim=-1)
    for layer_prf in watermarker.layer_prfs:
        g = (layer_prf.uniform(contexts, vocabulary) >= 0.5).to(probs.dtype)
        for row in range(probs.shape[0]):
            g_mass = (g[row] * probs[row]).sum()
            probs[row] = probs[row] * (1.0 + g[row] - g_mass)
    log_probs = torch.log(probs)
    return torch.where(
        torch.isfinite(log_probs), log_probs, torch.full_like(log_probs, -1e12)
    )


def _random_logits(batch: int, vocab: int, seed: int) -> torch.Tensor:
    return torch.randn(batch, vocab, generator=torch.Generator().manual_seed(seed))


def test_transform_matches_reference_implementation():
    watermarker = SynthIDWatermarker(KEY, CONTEXT_WIDTH, DEPTH)
    logits = _random_logits(3, VOCAB_SIZE, seed=0)
    contexts = torch.randint(
        0, VOCAB_SIZE, (3, CONTEXT_WIDTH), generator=torch.Generator().manual_seed(52)
    )
    expected = _naive_expected_g_update(logits, contexts, watermarker)
    actual = watermarker._transform_logits(logits, contexts)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_transform_preserves_truncated_support():
    """Tokens excluded by top-k/top-p must stay excluded after watermarking."""
    watermarker = SynthIDWatermarker(KEY, CONTEXT_WIDTH, DEPTH)
    logits = _random_logits(2, VOCAB_SIZE, seed=1)
    logits[:, VOCAB_SIZE // 2 :] = float("-inf")
    contexts = torch.randint(
        0, VOCAB_SIZE, (2, CONTEXT_WIDTH), generator=torch.Generator().manual_seed(63)
    )
    transformed = watermarker._transform_logits(logits, contexts)
    assert (transformed[:, VOCAB_SIZE // 2 :] <= -1e11).all()
    assert (transformed[:, : VOCAB_SIZE // 2] > -1e11).any()


def test_sparse_support_path_matches_reference():
    """With top-k truncation, the gathered-support path must produce the same
    update as the reference full-vocabulary computation."""
    watermarker = SynthIDWatermarker(KEY, CONTEXT_WIDTH, DEPTH)
    logits = _random_logits(3, VOCAB_SIZE, seed=11)
    top_k = 50
    cutoff = torch.topk(logits, top_k, dim=-1).values[:, -1:]
    logits = torch.where(
        logits < cutoff, torch.full_like(logits, float("-inf")), logits
    )
    contexts = torch.randint(
        0, VOCAB_SIZE, (3, CONTEXT_WIDTH), generator=torch.Generator().manual_seed(79)
    )
    expected = _naive_expected_g_update(logits, contexts, watermarker)
    actual = watermarker._transform_logits(logits, contexts)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_g_values_deterministic_and_balanced_across_layers():
    watermarker = SynthIDWatermarker(KEY, CONTEXT_WIDTH, DEPTH)
    contexts = torch.randint(
        0, VOCAB_SIZE, (64, CONTEXT_WIDTH), generator=torch.Generator().manual_seed(87)
    )
    vocabulary = torch.arange(VOCAB_SIZE)
    for layer_prf in watermarker.layer_prfs[:5]:
        uniforms = layer_prf.uniform(contexts, vocabulary)
        assert torch.equal(uniforms, layer_prf.uniform(contexts, vocabulary))
        fraction_ones = (uniforms >= 0.5).double().mean().item()
        assert abs(fraction_ones - 0.5) < 0.02


def test_layer_keys_produce_independent_g_values():
    watermarker = SynthIDWatermarker(KEY, CONTEXT_WIDTH, DEPTH)
    contexts = torch.randint(
        0, VOCAB_SIZE, (256, CONTEXT_WIDTH), generator=torch.Generator().manual_seed(98)
    )
    vocabulary = torch.arange(VOCAB_SIZE)
    g = torch.stack(
        [
            layer_prf.uniform(contexts, vocabulary) >= 0.5
            for layer_prf in watermarker.layer_prfs
        ]
    ).double()
    means = g.mean(dim=(1, 2), keepdim=True)
    centered = g - means
    covariance = torch.einsum("lbc,kbc->lk", centered, centered)
    covariance /= g.shape[1] * g.shape[2]
    variance = covariance.diagonal()
    correlation = covariance / torch.sqrt(torch.outer(variance, variance))
    off_diagonal = correlation[~torch.eye(DEPTH, dtype=torch.bool)]
    assert off_diagonal.abs().max().item() < 0.05


def test_watermarked_distribution_unbiased_in_expectation():
    """Non-distortion: averaging the watermarked distribution over independent
    g-value draws (fresh contexts) must recover the original distribution.

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


def test_detector_handles_empty_input():
    detector = SynthIDWatermarkDetector(KEY, CONTEXT_WIDTH, DEPTH)
    result = detector.detect([])
    assert result.num_scored_tokens == 0
    assert not result.is_watermarked


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        SynthIDWatermarker(KEY, context_width=0)
    with pytest.raises(ValueError):
        SynthIDWatermarker(KEY, depth=0)
    with pytest.raises(ValueError):
        SynthIDWatermarkDetector(KEY, depth=0)


def test_binomial_survival_matches_closed_form():
    # P(X >= k) for X ~ Binomial(4, 0.5) has an exact closed form.
    assert _binomial_survival(4, 4) == pytest.approx(1 / 16)
    assert _binomial_survival(3, 4) == pytest.approx(5 / 16)
    assert _binomial_survival(0, 4) == 1.0


def test_factory_creates_synthid_watermarker():
    config = WatermarkConfig(key=KEY, algorithm="synthid", synthid_depth=8)
    watermarker = create_watermarker(config)
    assert isinstance(watermarker, SynthIDWatermarker)
    assert watermarker.depth == 8
    assert len(watermarker.layer_prfs) == 8
