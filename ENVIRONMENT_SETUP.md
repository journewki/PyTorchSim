# PyTorchSim Environment Setup Guide

This document describes the steps required to set up the PyTorchSim environment
from scratch on **Ubuntu 22.04** with a conda-based Python environment.
It was written based on real debugging experience getting `tests/Llama/test_llama.py` to run.

---

## Problem Summary

Running any PyTorchSim test fails with:

```
RuntimeError: Expected one of cpu, cuda, ... device type at start of device string: npu
```

**Root cause:** `PyTorchSimDevice` (the C++ extension that registers the `npu` device
into PyTorch) was never built and installed. Several prerequisite issues must be
resolved before it can be built.

---

## Prerequisites

- Ubuntu 22.04
- conda environment with Python 3.10
- The `develop` branch of PyTorchSim (compatible with PyTorch 2.8)

---

## Step 1: Upgrade PyTorch to 2.8

The `develop` branch requires PyTorch 2.8+. The conda default (2.2.0) is incompatible.

```bash
pip install torch==2.8.0
```

> **Why:** The `develop` branch uses APIs introduced after PyTorch 2.2, including
> `at::HostAllocator`, `ATen/core/CachingHostAllocator.h`, `enable_gqa` in
> `_fused_sdp_choice`, and new virtual methods in `DeviceGuardImplInterface`.
> Building against 2.2.0 causes cascading C++ compilation errors.

---

## Step 2: Install CUDA Development Toolkit

The conda environment only includes CUDA runtime libraries, not the compiler (`nvcc`)
or development headers. These are required to build `PyTorchSimDevice`.

```bash
# Add NVIDIA apt repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update

# Install CUDA 12.1 compiler and development headers
apt-get install -y cuda-nvcc-12-1 cuda-cudart-dev-12-1 cuda-nvtx-12-1 libcublas-dev-12-1
```

Set environment variables:

```bash
export PATH=/usr/local/cuda-12.1/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.1
```

> **Why:** PyTorch's CMake config (`find_package(Torch)`) requires the full CUDA
> toolkit to be present — including `nvcc`, `nvToolsExt`, and cuBLAS headers —
> even though PyTorchSimDevice itself doesn't use CUDA directly.

---

## Step 3: Build TOGSim

TOGSim must be built before `PyTorchSimDevice`. It uses conan for dependency management.

```bash
cd /workspace/PyTorchSim/TOGSim
mkdir -p build && cd build
conan install .. --build=missing
cmake ..
make -j$(nproc)
```

Required tools: `conan==1.56.0`, `cmake>=3.18`

> **Why:** `PyTorchSimDevice` links against TOGSim's simulation infrastructure at runtime.

---

## Step 4: Install PyTorchSimDevice

```bash
export PATH=/usr/local/cuda-12.1/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.1
export TORCH_CUDA_ARCH_LIST="3.5;5.0;8.0;8.6;8.9;9.0"

cd /workspace/PyTorchSim/PyTorchSimDevice
pip install --no-build-isolation -e .
```

> **Why `--no-build-isolation`:** The build must use the system's already-installed
> PyTorch (and its CMake config), not a freshly isolated pip environment.
>
> **Why `TORCH_CUDA_ARCH_LIST`:** Without it, PyTorch's CMake auto-detects CUDA
> architectures and includes `9.0a` (Hopper TMA variant), which older CMake modules
> don't recognize, causing a fatal error.

---

## Step 5: Fix libstdc++ Version

The compiled `.so` requires `GLIBCXX_3.4.30`, which the conda `libstdc++` doesn't
provide. Use the system's newer version instead:

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

Add this to your `.bashrc` or run script to make it permanent.

> **Why:** The extension was compiled with GCC 11 (system), but conda ships an
> older `libstdc++`. Preloading the system library resolves the symbol version mismatch.

---

## Running Tests

With all steps complete, run tests as follows:

```bash
export PYTORCHSIM_ROOT_PATH=/workspace/PyTorchSim
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

cd /workspace/PyTorchSim
python tests/Llama/test_llama.py
```

---

## Quick Reference: All Exports

Add these to your shell profile (`~/.bashrc`) to avoid setting them every session:

```bash
export PYTORCHSIM_ROOT_PATH=/workspace/PyTorchSim
export PATH=/usr/local/cuda-12.1/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.1
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export TORCH_CUDA_ARCH_LIST="3.5;5.0;8.0;8.6;8.9;9.0"
```

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `Expected ... device type: npu` | `PyTorchSimDevice` not installed | Follow Steps 1–4 |
| `CUDA cannot be found` | No CUDA dev toolkit | Step 2 |
| `Unknown CUDA Architecture Name 9.0a` | Missing `TORCH_CUDA_ARCH_LIST` | Set env var in Step 4 |
| `ATen/core/CachingHostAllocator.h: No such file` | PyTorch version too old | Step 1: upgrade to 2.8 |
| `GLIBCXX_3.4.30 not found` | conda libstdc++ is outdated | Step 5 |
| `Failed to find nvToolsExt` | Missing cuda-nvtx package | Step 2 (install `cuda-nvtx-12-1`) |
