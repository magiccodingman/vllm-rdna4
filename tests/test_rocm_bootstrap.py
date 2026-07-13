# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace

from vllm import rocm_bootstrap


def test_amdsmi_bootstrap_only_runs_for_explicit_rocm(monkeypatch):
    calls = []
    fake_amdsmi = SimpleNamespace(amdsmi_init=lambda: calls.append("init"))
    monkeypatch.setitem(sys.modules, "amdsmi", fake_amdsmi)
    monkeypatch.setenv("VLLM_TARGET_DEVICE", "cuda")
    rocm_bootstrap._reset_amdsmi_bootstrap_for_tests()

    assert not rocm_bootstrap.initialize_amdsmi_for_rocm()
    assert calls == []


def test_amdsmi_bootstrap_is_idempotent(monkeypatch):
    calls = []
    fake_amdsmi = SimpleNamespace(amdsmi_init=lambda: calls.append("init"))
    monkeypatch.setitem(sys.modules, "amdsmi", fake_amdsmi)
    monkeypatch.setenv("VLLM_TARGET_DEVICE", "rocm")
    rocm_bootstrap._reset_amdsmi_bootstrap_for_tests()

    assert rocm_bootstrap.initialize_amdsmi_for_rocm()
    assert rocm_bootstrap.initialize_amdsmi_for_rocm()
    assert calls == ["init"]


def test_amdsmi_bootstrap_records_failure(monkeypatch):
    expected = RuntimeError("AMDSMI unavailable")

    def fail_init():
        raise expected

    monkeypatch.setitem(sys.modules, "amdsmi", SimpleNamespace(amdsmi_init=fail_init))
    monkeypatch.setenv("VLLM_TARGET_DEVICE", "rocm")
    rocm_bootstrap._reset_amdsmi_bootstrap_for_tests()

    assert not rocm_bootstrap.initialize_amdsmi_for_rocm()
    assert rocm_bootstrap.get_amdsmi_bootstrap_error() is expected
