# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Early ROCm bootstrap helpers that must run before importing PyTorch."""

from __future__ import annotations

import os

_AMDSMI_INITIALIZED = False
_AMDSMI_INIT_ERROR: Exception | None = None


def initialize_amdsmi_for_rocm() -> bool:
    """Initialize AMDSMI before PyTorch when ROCm is explicitly requested.

    The TheRock ROCm preview can require AMDSMI to be initialized before the
    HIP runtime is first touched. Initialization is best-effort so ordinary
    ROCm installations without AMDSMI keep the existing detection path.
    """
    global _AMDSMI_INITIALIZED, _AMDSMI_INIT_ERROR

    if _AMDSMI_INITIALIZED:
        return True
    if os.environ.get("VLLM_TARGET_DEVICE", "").strip().lower() != "rocm":
        return False

    try:
        import amdsmi

        amdsmi.amdsmi_init()
    except Exception as exc:  # Import-time bootstrap cannot safely log yet.
        _AMDSMI_INIT_ERROR = exc
        return False

    _AMDSMI_INITIALIZED = True
    _AMDSMI_INIT_ERROR = None
    return True


def get_amdsmi_bootstrap_error() -> Exception | None:
    return _AMDSMI_INIT_ERROR


def _reset_amdsmi_bootstrap_for_tests() -> None:
    global _AMDSMI_INITIALIZED, _AMDSMI_INIT_ERROR
    _AMDSMI_INITIALIZED = False
    _AMDSMI_INIT_ERROR = None
