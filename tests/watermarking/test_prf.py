# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.watermarking.prfs import PhiloxPRF


def test_philox_compatibility_vector():
    contexts = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7]])
    token_ids = torch.tensor([7, 8])
    expected_mantissas = [[1661833, 10571167], [9742331, 13375724]]
    expected = (
        (torch.tensor(expected_mantissas, dtype=torch.float64) + 1) / (2**24 + 1)
    ).to(torch.float32)

    assert torch.equal(PhiloxPRF(42).uniform(contexts, token_ids), expected)


def test_prf_pairs_contexts_with_target_tokens():
    contexts = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7]])
    token_ids = torch.tensor([[7], [8]])

    assert PhiloxPRF(42).uniform(contexts, token_ids).shape == (2, 1)


def test_philox_bits_word_count_and_determinism():
    contexts = torch.randint(-1, 150000, (8, 4))
    token_ids = torch.randint(0, 150000, (8, 64))
    prf = PhiloxPRF(42)
    words = prf.bits(contexts, token_ids, 2)
    assert len(words) == 2
    assert all(word.shape == (8, 64) for word in words)
    again = prf.bits(contexts, token_ids, 2)
    assert all(torch.equal(a, b) for a, b in zip(words, again, strict=True))
    with pytest.raises(ValueError):
        prf.bits(contexts, token_ids, 0)
    with pytest.raises(ValueError):
        prf.bits(contexts, token_ids, 5)


def test_philox_bits_are_balanced():
    contexts = torch.randint(-1, 150000, (64, 4))
    token_ids = torch.randint(0, 150000, (64, 512))
    words = PhiloxPRF(42).bits(contexts, token_ids, 4)
    shifts = torch.arange(32)
    for word in words:
        bits = (word.unsqueeze(-1) >> shifts) & 1
        assert abs(bits.to(torch.float64).mean().item() - 0.5) < 0.02


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
@pytest.mark.parametrize("key", [42, 15726070495360670683])
@pytest.mark.parametrize("context_width", [1, 4, 16])
@pytest.mark.parametrize("num_words", [1, 4])
def test_philox_bits_accelerator_matches_cpu(
    key: int, context_width: int, num_words: int
):
    contexts = torch.arange(2 * context_width, dtype=torch.int64).reshape(
        2, context_width
    )
    token_ids = torch.arange(1024, dtype=torch.int64)
    prf = PhiloxPRF(key)

    expected = prf.bits(contexts, token_ids, num_words)
    actual = prf.bits(contexts.cuda(), token_ids.cuda(), num_words)

    for expected_word, actual_word in zip(expected, actual, strict=True):
        torch.testing.assert_close(actual_word.cpu(), expected_word, rtol=0, atol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
@pytest.mark.parametrize("key", [42, 15726070495360670683])
@pytest.mark.parametrize("context_width", [1, 4, 16])
def test_philox_accelerator_matches_cpu(key: int, context_width: int):
    contexts = torch.arange(2 * context_width, dtype=torch.int64).reshape(
        2, context_width
    )
    token_ids = torch.arange(1024, dtype=torch.int64)
    prf = PhiloxPRF(key)

    expected = prf.uniform(contexts, token_ids)
    actual = prf.uniform(contexts.cuda(), token_ids.cuda()).cpu()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
