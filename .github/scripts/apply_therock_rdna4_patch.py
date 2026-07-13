from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace_exact(relative_path: str, old: str, new: str, *, count: int = 1) -> None:
    path = ROOT / relative_path
    text = path.read_text()
    actual = text.count(old)
    if actual != count:
        raise RuntimeError(
            f"{relative_path}: expected {count} occurrence(s) of patch anchor, "
            f"found {actual}"
        )
    path.write_text(text.replace(old, new, count))


def write(relative_path: str, content: str) -> None:
    path = ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


write(
    "vllm/rocm_bootstrap.py",
    '''# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Early ROCm bootstrap helpers that must run before importing PyTorch."""

from __future__ import annotations

import os

_AMDSMI_INITIALIZED = False
_AMDSMI_INIT_ERROR: BaseException | None = None


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
    except BaseException as exc:  # Import-time bootstrap cannot safely log yet.
        _AMDSMI_INIT_ERROR = exc
        return False

    _AMDSMI_INITIALIZED = True
    _AMDSMI_INIT_ERROR = None
    return True


def get_amdsmi_bootstrap_error() -> BaseException | None:
    return _AMDSMI_INIT_ERROR


def _reset_amdsmi_bootstrap_for_tests() -> None:
    global _AMDSMI_INITIALIZED, _AMDSMI_INIT_ERROR
    _AMDSMI_INITIALIZED = False
    _AMDSMI_INIT_ERROR = None
''',
)

replace_exact(
    "vllm/__init__.py",
    "from .version import __version__, __version_tuple__  # isort:skip\n\nimport typing\n",
    "from .version import __version__, __version_tuple__  # isort:skip\n"
    "from .rocm_bootstrap import initialize_amdsmi_for_rocm  # isort:skip\n\n"
    "# TheRock ROCm preview builds may require AMDSMI initialization before\n"
    "# env_override imports torch and touches the HIP runtime.\n"
    "initialize_amdsmi_for_rocm()\n\n"
    "import typing\n",
)

replace_exact(
    "vllm/_aiter_ops.py",
    "import vllm.envs as envs\nfrom vllm.platforms import current_platform\n",
    "import vllm.envs as envs\nfrom vllm.logger import init_logger\n"
    "from vllm.platforms import current_platform\n",
)
replace_exact(
    "vllm/_aiter_ops.py",
    "except ImportError:\n    pd = PlaceholderModule(\"pandas\")\n\n# fp8_dtype is not cached.\n",
    "except ImportError:\n    pd = PlaceholderModule(\"pandas\")\n\n"
    "logger = init_logger(__name__)\n\n"
    "# fp8_dtype is not cached.\n",
)
replace_exact(
    "vllm/_aiter_ops.py",
    '''    if current_platform.is_rocm() and IS_AITER_FOUND:
        from vllm.platforms.rocm import on_mi3xx

        return on_mi3xx()
    return False
''',
    '''    if current_platform.is_rocm() and IS_AITER_FOUND:
        from vllm.platforms.rocm import on_gfx12x, on_mi3xx

        return on_mi3xx() or on_gfx12x()
    return False


def is_aiter_sampler_supported() -> bool:
    """Whether the installed AITER sampler can run on this architecture.

    AITER supports a growing set of gfx12 kernels, but its sampling JIT still
    rejects gfx1201. Keep the rest of AITER available on RDNA4 while selecting
    vLLM's native sampler until that upstream kernel gains gfx12 support.
    """
    if not is_aiter_found_and_supported():
        return False
    from vllm.platforms.rocm import on_gfx12x

    return not on_gfx12x()
''',
)
replace_exact(
    "vllm/_aiter_ops.py",
    "    Checks: platform (ROCm), device arch (gfx9), and library existence.\n",
    "    Checks: platform (ROCm), supported device arch (MI3xx or gfx12x), and library existence.\n",
)
replace_exact(
    "vllm/_aiter_ops.py",
    '''    except Exception:
        return set()
''',
    '''    except Exception as exc:
        logger.debug_once(
            "Unable to load AITER tuned GEMM configurations from %s: %r. "
            "The affected kernels will fall back to another implementation.",
            csv_path,
            exc,
        )
        return set()
''',
)
replace_exact(
    "vllm/_aiter_ops.py",
    "    ROCm AITER package is supported and enabled on gfx9 archs.\n",
    "    ROCm AITER package is supported on the current AMD architecture.\n",
)
replace_exact(
    "vllm/_aiter_ops.py",
    '''    @classmethod
    @if_aiter_supported
    def is_enabled(cls) -> bool:
        return cls._AITER_ENABLED

    @classmethod
    @if_aiter_supported
    def is_linear_enabled(cls) -> bool:
''',
    '''    @classmethod
    @if_aiter_supported
    def is_enabled(cls) -> bool:
        return cls._AITER_ENABLED

    @classmethod
    @if_aiter_supported
    def is_sampler_enabled(cls) -> bool:
        return cls._AITER_ENABLED and is_aiter_sampler_supported()

    @classmethod
    @if_aiter_supported
    def is_linear_enabled(cls) -> bool:
''',
)
replace_exact(
    "vllm/_aiter_ops.py",
    "        which verifies: (1) platform is ROCm, (2) device arch is gfx9, and\n",
    "        which verifies: (1) platform is ROCm, (2) device arch is MI3xx or gfx12x, and\n",
)
replace_exact(
    "vllm/_aiter_ops.py",
    "                        current_platform.is_on_gfx9() and   | @if_aiter_supported\n",
    "                        supported ROCm architecture and     | @if_aiter_supported\n",
)

replace_exact(
    "vllm/v1/sample/ops/topk_topp_sampler.py",
    "            and rocm_aiter_ops.is_enabled()\n",
    "            and rocm_aiter_ops.is_sampler_enabled()\n",
)

replace_exact(
    "vllm/platforms/rocm.py",
    '''    backends = []
    # Keep ROCM_ATTN disabled for KV connectors until connector transfer
    # semantics are validated for its asymmetric native K/V cache views.
    if not use_kv_connector:
        backends.append(AttentionBackendEnum.ROCM_ATTN)
    if rocm_aiter_ops.is_mha_enabled():
        backends.append(AttentionBackendEnum.ROCM_AITER_FA)
    if is_aiter_found_and_supported():
        backends.append(AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN)
    backends.append(AttentionBackendEnum.TRITON_ATTN)
''',
    '''    backends = []
    prefer_aiter_unified = bool(rocm_aiter_ops.is_triton_unified_attn_enabled())
    if prefer_aiter_unified:
        backends.append(AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN)

    # Keep ROCM_ATTN disabled for KV connectors until connector transfer
    # semantics are validated for its asymmetric native K/V cache views.
    if not use_kv_connector:
        backends.append(AttentionBackendEnum.ROCM_ATTN)
    if rocm_aiter_ops.is_mha_enabled():
        backends.append(AttentionBackendEnum.ROCM_AITER_FA)
    if is_aiter_found_and_supported() and not prefer_aiter_unified:
        backends.append(AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN)
    backends.append(AttentionBackendEnum.TRITON_ATTN)
''',
)
replace_exact(
    "vllm/platforms/rocm.py",
    '''    @classmethod
    def use_custom_allreduce(cls) -> bool:
        # We only enable custom allreduce for MI300 series
        return any(gfx in _GCN_ARCH for gfx in ["gfx94", "gfx95"])
''',
    '''    @classmethod
    def use_custom_allreduce(cls) -> bool:
        if on_mi3xx():
            return True
        if not on_gfx12x():
            return False
        if not (
            envs.VLLM_ROCM_USE_AITER
            and envs.VLLM_ROCM_USE_AITER_CUSTOM_AR
        ):
            return False

        from importlib.util import find_spec

        return find_spec("aiter") is not None
''',
)

replace_exact(
    "vllm/distributed/device_communicators/cuda_communicator.py",
    '''        if use_custom_allreduce and self.world_size > 1 and current_platform.is_rocm():
            # Initialize a custom quick all-reduce implementation for AMD.
            # Quick reduce is designed as a complement to custom allreduce
            # (vLLM's or AITER's), so it is initialized for either backend.
            # Based on quickreduce (https://github.com/mk1-project/quickreduce).
            # On ROCm, 'use_custom_allreduce==True' means it must currently be
            # an MI300 series.
            self.qr_comm = QuickAllReduce(group=self.cpu_group, device=self.device)
''',
    '''        if use_custom_allreduce and self.world_size > 1 and current_platform.is_rocm():
            # QuickReduce remains MI3xx-only. gfx12 may enable the generic
            # custom-allreduce switch solely to construct AITER CustomAllreduce.
            from vllm.platforms.rocm import on_mi3xx

            if on_mi3xx():
                self.qr_comm = QuickAllReduce(
                    group=self.cpu_group,
                    device=self.device,
                )
''',
)

write(
    "tests/test_rocm_bootstrap.py",
    '''# SPDX-License-Identifier: Apache-2.0
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
''',
)

write(
    "tests/rocm/test_therock_rdna4_compat.py",
    '''# SPDX-License-Identifier: Apache-2.0
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
''',
)

write(
    "docs/getting_started/installation/gpu/rocm_therock_rdna4.md",
    '''# TheRock ROCm preview on RDNA4

> **Experimental:** This path targets AMD's TheRock preview userspace on
> gfx12/RDNA4 devices such as the Radeon AI PRO R9700. Keep the ROCm runtime,
> PyTorch, Triton, AITER, and vLLM build aligned to the same preview stack.

## What vLLM handles

When `VLLM_TARGET_DEVICE=rocm` is set, vLLM initializes AMDSMI before importing
PyTorch. This replaces the external Python wrapper previously needed to avoid
`Failed to infer device type` during startup on TheRock builds.

AITER is recognized on gfx12, but individual kernels keep their own capability
checks. The current AITER sampling JIT rejects gfx1201, so vLLM uses its native
sampler while leaving compatible AITER paths available.

Setting `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1` makes unified attention the
first automatic ROCm attention candidate. AITER custom all-reduce can be used
on gfx12 when both the parent AITER switch and
`VLLM_ROCM_USE_AITER_CUSTOM_AR` are enabled. QuickReduce remains MI3xx-only.

## Environment

Use a matching preview software stack. A second Triton or `triton-kernels`
installation from a different ROCm release can silently change kernel
selection or break JIT compilation. Confirm package versions and import paths
before benchmarking.

```bash
python - <<'PY'
import importlib.metadata as md
import torch
import triton

print("torch:", torch.__version__)
print("HIP:", torch.version.hip)
print("triton:", md.version("triton"), triton.__file__)
try:
    print("triton-kernels:", md.version("triton-kernels"))
except md.PackageNotFoundError:
    print("triton-kernels: not installed")
print("GPU:", torch.cuda.get_device_name(0))
print("arch:", torch.cuda.get_device_properties(0).gcnArchName)
PY
```

## Example launch

```bash
MODEL=/path/to/model

HIP_VISIBLE_DEVICES=0,1 \\
ROCR_VISIBLE_DEVICES=0,1 \\
VLLM_TARGET_DEVICE=rocm \\
VLLM_ROCM_USE_AITER=1 \\
VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 \\
VLLM_ROCM_USE_AITER_RMSNORM=1 \\
VLLM_ROCM_USE_AITER_TRITON_GEMM=1 \\
VLLM_ROCM_USE_AITER_CUSTOM_AR=1 \\
vllm serve "$MODEL" \\
  --tensor-parallel-size 2 \\
  --attention-backend ROCM_AITER_UNIFIED_ATTN
```

Variables such as `HSA_ENABLE_SDMA=0` and `HSA_FORCE_FINE_GRAIN_PCIE=1` are
platform workarounds, not universal performance defaults. Benchmark them both
on and off for the exact driver build and topology.

## Validation

At startup, verify the intended attention backend, RMSNorm priority, FP8 linear
kernel, and all-reduce backends. A successful AITER import does not prove every
operation selected an AITER kernel; tuned GEMM configuration misses fall back
and are logged at debug level.
''',
)

for relative_path in (
    "vllm/__init__.py",
    "vllm/rocm_bootstrap.py",
    "vllm/_aiter_ops.py",
    "vllm/platforms/rocm.py",
    "vllm/v1/sample/ops/topk_topp_sampler.py",
    "vllm/distributed/device_communicators/cuda_communicator.py",
    "tests/test_rocm_bootstrap.py",
    "tests/rocm/test_therock_rdna4_compat.py",
):
    compile((ROOT / relative_path).read_text(), relative_path, "exec")

print("TheRock RDNA4 compatibility patch applied and syntax-checked.")
