# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rules
For simple tasks such as web search, paper reading, code reading, call Sonnet as subagents. 
For complex tasks such as planning and coding and debugging codes, call Opus as subagents.
Always before making any change, search on the web for the newest documentation.
Only implement if you are 100% sure it will work. 


## Build Commands

### TOGSim (C++ simulator backend)
```bash
cd TOGSim
mkdir -p build && cd build
conan install .. --build=missing   # requires conan==1.56.0
cmake ..
make -j$(nproc)
```

### PyTorchSimDevice (PyTorch custom device extension)
Must be built after TOGSim. Requires CUDA 12.1 dev toolkit (not just conda runtime).
```bash
export PATH=/usr/local/cuda-12.1/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.1
export TORCH_CUDA_ARCH_LIST="3.5;5.0;8.0;8.6;8.9;9.0"
cd PyTorchSimDevice
pip install --no-build-isolation -e .
```

## Running Tests

```bash
cd /workspace/PyTorchSim
python tests/test_matmul.py         # single op test
python tests/test_scheduler.py      # multi-tenancy test
python tests/Llama/test_llama.py    # model test
```

Results are written to `$TORCHSIM_LOG_PATH/<hash>/togsim_result/`.

## Required Environment Variables

```bash
export TORCHSIM_DIR=/workspace/PyTorchSim
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6  # fix GLIBCXX_3.4.30 mismatch
```

Optional:
```bash
export TOGSIM_CONFIG=/workspace/PyTorchSim/configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml
export pytorchsim_functional_mode=False   # skip Spike ISA verification for speed
export TORCHSIM_LOG_PATH=/tmp/torchinductor
export SRAM_BUFFER_PLAN_PATH=tpuv4/gemm_plan.py  # L2 persistent cache allocation plan
```

## Architecture Overview

PyTorchSim has two major components that work together:

### 1. Compiler (`PyTorchSimFrontend/`)
Hooks into `torch.compile()` as a custom inductor backend. When a model runs on the `npu` device, it:
1. Lowers PyTorch ops to MLIR via `mlir_codegen_backend.py`
2. Compiles MLIR → RISC-V binary (via LLVM) + Tile-Operation Graph (TOG, a `.tog` file)
3. The TOG encodes the sequence of hardware tiles (GEMM tiles, vector tiles, etc.) with dependencies

Key files: `extension_config.py` (hardware config), `mlir_ops.py` (op lowering), `mlir_gemm_template.py` / `mlir_conv_*.py` (tile templates).

### 2. TOGSim (`TOGSim/`)
C++ cycle-accurate simulator that executes TOGs. Three simulation phases run per kernel:
1. **Gem5**: micro-architecture timing for each tile
2. **Spike** (optional): RISC-V ISA functional verification
3. **TOGSim**: full cycle-accurate simulation with DRAM (Ramulator2/simple), NoC (BookSim2/simple), and L2 cache

Python wrapper: `Simulator/simulator.py` (`TOGSimulator` context manager, `FunctionalSimulator` for Spike).

### 3. PyTorchSimDevice (`PyTorchSimDevice/`)
C++ PyTorch extension that registers `npu` as a custom device type using PyTorch's `PrivateUse1` / `torch_openreg` mechanism. Without this installed, any `torch.device("npu:0")` call fails. It implements device memory allocation, streams, and device guards.

### 4. Scheduler (`Scheduler/scheduler.py`)
Python-level inference serving simulator. Manages incoming `Request` objects (with Poisson-distributed arrival times), batches them by model, and dispatches compiled models through `PyTorchSimRunner`. Used only for multi-tenancy experiments, not for single-model tests.

### Data Flow for a Single Test
```
torch.compile(model)(input on npu:0)
  → PyTorchSimFrontend generates RISC-V binary + .tog file
  → Simulator/simulator.py calls Gem5 → Spike → TOGSim
  → TOGSim/src/Scheduler.cc dispatches tiles to cores cycle by cycle
  → Stats logged to TORCHSIM_LOG_PATH
```

## Configuration Files (`configs/`)

YAML files define NPU hardware. Key parameters:
- `num_cores`, `core_freq_mhz`, `num_systolic_array_per_core`
- `vpu_num_lanes`, `vpu_spad_size_kb_per_lane` — vector processing unit
- `dram_type` (`ramulator2` or `simple`), `dram_channels`
- `icnt_type` (`booksim` or `simple`) — network-on-chip
- `codegen_mapping_strategy` — `heuristic` (default), `autotune`, or `external`
- `codegen_compiler_optimization` — `"all"`, `"none"`, or list of specific optimizations
- `num_partition` + `partition` — for multi-core partitioning (multi-tenancy)

## Test Patterns

Most tests follow this pattern:
```python
device = torch.device("npu:0")
opt_fn = torch.compile(dynamic=False)(model_or_fn)
result = opt_fn(input.to(device))
cpu_result = model_or_fn(input.to("cpu"))
test_result("Name", result, cpu_result)  # checks correctness
```

Tests use `dynamic=False` (static shapes) because the compiler generates shape-specific tile mappings.

For multi-tenancy, see `tests/test_scheduler.py` and `tests/test_scheduler_batching.py`.

## Branch Convention
- `develop` branch: active development, requires PyTorch 2.8+
- `master` branch: stable, targets PyTorch 2.2 (see README — note `develop` is ahead)

See `ENVIRONMENT_SETUP.md` for full troubleshooting guide on building from scratch.
