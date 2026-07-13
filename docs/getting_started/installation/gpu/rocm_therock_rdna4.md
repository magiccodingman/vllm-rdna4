# TheRock ROCm preview on RDNA4

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

HIP_VISIBLE_DEVICES=0,1 \
ROCR_VISIBLE_DEVICES=0,1 \
VLLM_TARGET_DEVICE=rocm \
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 \
VLLM_ROCM_USE_AITER_RMSNORM=1 \
VLLM_ROCM_USE_AITER_TRITON_GEMM=1 \
VLLM_ROCM_USE_AITER_CUSTOM_AR=1 \
vllm serve "$MODEL" \
  --tensor-parallel-size 2 \
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
