# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util

import pytest

import vllm._aiter_ops as aiter_ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms import rocm

pytestmark = pytest.mark.skipif(
    not aiter_ops.current_platform.is_rocm(),
    reason="ROCm-only compatibility tests",
)


def test_gfx12_is_aiter_capable_but_sampler_stays_disabled(monkeypatch):
    monkeypatch.setattr(aiter_ops, "IS_AITER_FOUND", True)
    monkeypatch.setattr(rocm, "_ON_MI3XX", False)
    monkeypatch.setattr(rocm, "_ON_GFX12X", True)
    monkeypatch.setattr(rocm_aiter_ops, "_AITER_ENABLED", True)

    assert aiter_ops.is_aiter_found_and_supported()
    assert rocm_aiter_ops.is_enabled()
    assert not rocm_aiter_ops.is_sampler_enabled()


def test_gfx12_custom_ar_requires_aiter_opt_in(monkeypatch):
    monkeypatch.setattr(rocm, "_ON_MI3XX", False)
    monkeypatch.setattr(rocm, "_ON_GFX12X", True)
    monkeypatch.setattr(rocm.envs, "VLLM_ROCM_USE_AITER", True)
    monkeypatch.setattr(rocm.envs, "VLLM_ROCM_USE_AITER_CUSTOM_AR", True)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())

    assert rocm.RocmPlatform.use_custom_allreduce()

    monkeypatch.setattr(rocm.envs, "VLLM_ROCM_USE_AITER_CUSTOM_AR", False)
    assert not rocm.RocmPlatform.use_custom_allreduce()
