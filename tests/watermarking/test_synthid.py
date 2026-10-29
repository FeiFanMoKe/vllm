# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config.watermarking import WatermarkConfig
from vllm.platforms import current_platform
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
            probs[row] = probs[row] * (1.0 + watermarker.strength * (g[row] - g_mass))
    log_probs = torch.log(probs)
    return torch.where(
        torch.isfinite(logits),
        log_probs,
        torch.full_like(log_probs, float("-inf")),
    )


def _random_logits(batch: int, vocab: int, seed: int) -> torch.Tensor:
    return torch.randn(batch, vocab, generator=torch.Generator().manual_seed(seed))


def _eager_update(
    logp: torch.Tensor,
    words: torch.Tensor,
    depth: int,
    strength: float = 1.0,
) -> torch.Tensor:
    for layer in range(depth):
        g = ((words >> (layer % 32)) & 1).to(logp.dtype)
        g_mass = (g * logp.exp()).sum(dim=-1, keepdim=True)
        logp = logp + torch.log1p(strength * (g - g_mass))
    return logp


def _assert_update_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    # The 30-layer multiplicative update amplifies tiny math-library
    # differences on tail tokens; the sampling distribution is what matters.
    torch.testing.assert_close(
        actual.softmax(dim=-1),
        expected.softmax(dim=-1),
        atol=1e-7,
        rtol=1e-5,
    )
    finite = torch.isfinite(expected)
    torch.testing.assert_close(actual[finite], expected[finite], atol=5e-2, rtol=0)
    assert torch.equal(torch.isfinite(actual), torch.isfinite(expected))


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


def test_transform_matches_reference_with_strength():
    watermarker = SynthIDWatermarker(KEY, CONTEXT_WIDTH, DEPTH, strength=0.5)
    logits = _random_logits(3, VOCAB_SIZE, seed=21)
    contexts = _random_contexts(3, seed=21)
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


def test_max_probability_gate_skips_peaked_rows():
    watermarker = SynthIDWatermarker(KEY, CONTEXT_WIDTH, DEPTH, max_probability=0.5)
    logits = _random_logits(2, 128, seed=22)
    logits[0, 0] = 20.0
    contexts = _random_contexts(2, seed=22)
    mixed = watermarker._mixed_logits(logits, contexts, None)
    assert torch.equal(mixed[0], logits[0])
    assert not torch.equal(mixed[1], logits[1])


def test_max_probability_one_does_not_gate():
    watermarker = SynthIDWatermarker(KEY, CONTEXT_WIDTH, DEPTH)
    logits = _random_logits(1, 128, seed=23)
    logits[0, 0] = 20.0
    contexts = _random_contexts(1, seed=23)
    mixed = watermarker._mixed_logits(logits, contexts, None)
    assert not torch.equal(mixed[0], logits[0])


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
        SynthIDWatermarker(KEY, strength=0.0)
    with pytest.raises(ValueError):
        SynthIDWatermarker(KEY, strength=1.5)
    with pytest.raises(ValueError):
        SynthIDWatermarker(KEY, max_probability=0.0)
    with pytest.raises(ValueError):
        SynthIDWatermarker(KEY, max_probability=1.5)
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
    config = WatermarkConfig(
        key=KEY,
        algorithm="synthid",
        synthid_depth=8,
        synthid_strength=0.5,
        synthid_max_probability=0.75,
    )
    watermarker = create_watermarker(config)
    assert isinstance(watermarker, SynthIDWatermarker)
    assert watermarker.depth == 8
    assert watermarker.strength == 0.5
    assert watermarker.max_probability == 0.75


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
@pytest.mark.parametrize(
    ("batch", "candidates"), [(1, 50), (4, 257), (8, 1024), (2, 4096)]
)
def test_fused_update_matches_eager(batch: int, candidates: int):
    from vllm.v1.worker.gpu.sample.watermark import synthid_update

    generator = torch.Generator(device="cuda").manual_seed(0)
    logits = torch.randn(batch, candidates, device="cuda", generator=generator)
    logp = logits.log_softmax(dim=-1)
    words = torch.randint(
        -(2**31),
        2**31,
        (batch, candidates),
        dtype=torch.int64,
        device="cuda",
        generator=generator,
    )
    expected = _eager_update(logp, words, DEPTH)
    actual = synthid_update(logp, words, DEPTH)
    _assert_update_close(actual, expected)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
@pytest.mark.parametrize("strength", [0.25, 0.5])
def test_fused_update_matches_eager_with_strength(strength: float):
    from vllm.v1.worker.gpu.sample.watermark import synthid_update

    generator = torch.Generator(device="cuda").manual_seed(2)
    logits = torch.randn(2, 256, device="cuda", generator=generator)
    logp = logits.log_softmax(dim=-1)
    words = torch.randint(
        -(2**31),
        2**31,
        (2, 256),
        dtype=torch.int64,
        device="cuda",
        generator=generator,
    )
    expected = _eager_update(logp, words, DEPTH, strength)
    actual = synthid_update(logp, words, DEPTH, strength)
    _assert_update_close(actual, expected)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
def test_fused_update_handles_masked_candidates():
    from vllm.v1.worker.gpu.sample.watermark import synthid_update

    generator = torch.Generator(device="cuda").manual_seed(1)
    logits = torch.randn(2, 128, device="cuda", generator=generator)
    logits[:, 64:] = float("-inf")
    logp = logits.log_softmax(dim=-1)
    words = torch.randint(
        -(2**31),
        2**31,
        (2, 128),
        dtype=torch.int64,
        device="cuda",
        generator=generator,
    )
    expected = _eager_update(logp, words, DEPTH)
    actual = synthid_update(logp, words, DEPTH)
    _assert_update_close(actual, expected)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
def test_transform_cuda_matches_cpu():
    watermarker = SynthIDWatermarker(KEY, CONTEXT_WIDTH, DEPTH)
    logits = _random_logits(2, VOCAB_SIZE, seed=11)
    contexts = _random_contexts(2, seed=11)
    expected = watermarker._transform_logits(logits, contexts)
    actual = watermarker._transform_logits(logits.cuda(), contexts.cuda()).cpu()
    _assert_update_close(actual, expected)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
def test_transform_deep_cuda_falls_back():
    watermarker = SynthIDWatermarker(KEY, CONTEXT_WIDTH, 64)
    logits = _random_logits(2, VOCAB_SIZE, seed=12)
    contexts = _random_contexts(2, seed=12)
    expected = watermarker._transform_logits(logits, contexts)
    actual = watermarker._transform_logits(logits.cuda(), contexts.cuda()).cpu()
    _assert_update_close(actual, expected)
