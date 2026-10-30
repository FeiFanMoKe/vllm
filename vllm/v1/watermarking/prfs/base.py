# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod

import torch

_UINT32_MASK = 2**32 - 1


class WatermarkPRF(ABC):
    supports_bits: bool = False

    @abstractmethod
    def uniform(self, contexts: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def bits(
        self,
        contexts: torch.Tensor,
        token_ids: torch.Tensor,
        num_words: int = 4,
    ) -> tuple[torch.Tensor, ...]:
        """Packed g-value bits as ``num_words`` uint32 words held in int64.

        Bit ``i`` of the concatenated words is the ``i``-th layer's g-value.
        One evaluation serves every layer, as in the SynthID-Text reference
        implementation.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide packed g-value bits"
        )


def uint32_to_uniform(values: torch.Tensor) -> torch.Tensor:
    mantissas = (values & _UINT32_MASK) >> 8
    return ((mantissas.to(torch.float64) + 1.0) / (2**24 + 1)).to(torch.float32)
