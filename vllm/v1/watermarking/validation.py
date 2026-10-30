# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import warnings


def validate_context_width(context_width: int) -> None:
    if context_width < 1:
        raise ValueError("context_width must be positive")
    if context_width > 16:
        warnings.warn(
            "context_width values greater than 16 reduce robustness to edits because "
            "each changed token affects more subsequent watermark contexts",
            stacklevel=3,
        )
