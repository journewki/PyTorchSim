# Experiment Log: samsung_mx_npu.yml + DDR5 DRAM Debug
**Date:** 2026-03-19
**Branch:** dit_training
**Test:** `tests/Llama/test_llama.py` with `configs/samsung_mx_npu.yml`

---

## Goal
Run Llama decoder layer test with a custom Samsung MX NPU config using DDR5-4800 DRAM.

---

## Problems & Fixes

### Problem 1: GLIBCXX_3.4.30 not found
**Error:**
```
ImportError: libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by libopenreg.so)
```
**Cause:** conda's `libstdc++.so.6` is older than the version `libopenreg.so` was compiled against.
**Fix:** Set `LD_PRELOAD` to use the system's newer libstdc++:
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```
Add to `~/.bashrc` to avoid setting it every time.

---

### Problem 2: Ramulator2 Frontend `MessReqGenerator` not registered
**Error:**
```
Trying to create an implementation "MessReqGenerator" of interface "Frontend", but the implementation is not registered!
```
**Cause:** `configs/ramulator2_configs/ddr5_4800an_16ch.yaml` was taken from the Ramulator2 repo's `mess` branch, which uses `MessReqGenerator` as a standalone traffic generator frontend. TOGSim's Ramulator2 build only registers the `GEM5` frontend (since TOGSim drives Ramulator2 directly).
**Fix:** Change the `Frontend` section in `ddr5_4800an_16ch.yaml`:
```yaml
# Before
Frontend:
  impl: MessReqGenerator
  start_addr: 0
  ...

# After
Frontend:
  impl: GEM5
```

---

### Problem 3: DDR5_4800AN timing preset not recognized
**Error:**
```
Unrecognized timing preset "DDR5_4800AN" in DDR5!
```
**Cause:** TOGSim's Ramulator2 is based on the `main` branch, which only has DDR5-3200 presets (`DDR5_3200AN`, `DDR5_3200BN`, `DDR5_3200C`). The `DDR5_4800AN` preset only exists in the Ramulator2 `mess` branch.
**Fix:** Added `DDR5_4800AN` timing entry to `TOGSim/extern/ramulator2/src/dram/impl/DDR5.cpp` (values taken from Ramulator2 mess branch):
```cpp
{"DDR5_4800AN", {4800, 8, 34, 34, 34, 77, 111, 72, 18, 32, 2, 8, 8, 32+8+6, 12, 48, 32+8+24, 8, -1, -1, -1, -1, -1, -1, 73, -1, -1, -1, -1, -1, 2, 416}},
```

---

### Problem 4: `nRRDL` timing not specified (SIGABRT)
**Error:**
```
In "DDR5", timing nRRDL is not specified!
```
**Cause:** TOGSim's DDR5.cpp has lookup tables (`nRRDL_TABLE`, `nFAW_TABLE`, `nCCD_L_WR2_TABLE`) that only cover DDR5-3200. When rate=4800, `rate_id` returned -1, leaving `nRRDL` unset (-1), which triggers an assertion.
**Fix:** Extended the lookup tables in `TOGSim/extern/ramulator2/src/dram/impl/DDR5.cpp` to include DDR5-4800 (values from Ramulator2 mess branch):
```cpp
// rate_id switch
case 4800: return 1;

// nRRDL_TABLE [3][2]  (was [3][1])
// 3200  4800
  { 5,   12 },  // x4
  { 5,   12 },  // x8
  { 5,   12 },  // x16

// nFAW_TABLE [3][2]  (was [3][1])
// 3200  4800
  { 40,  40 },  // x4
  { 32,  32 },  // x8
  { 32,  32 },  // x16

// nCCD_L_WR2_TABLE [2]  (was [1])
// 3200  4800
  32,    32
```

---

### Problem 5: Pre-existing compile error in Simulator.cc
**Error:**
```
error: 'Number' was not declared in this scope
  _noc_node_per_core = config.icnt_injection_ports_per_core; Number of NoC injection ports per core
```
**Cause:** A comment was written without `//`, causing the compiler to treat it as code.
**Fix:** Added missing `//` comment markers in `TOGSim/src/Simulator.cc`:
```cpp
// Before
_noc_node_per_core = config.icnt_injection_ports_per_core; Number of NoC injection ports per core
char* onnxim_path_env = std::getenv("TORCHSIM_DIR"); // Path of onnxim(== TOGSim)
std::string onnxim_path = onnxim_path_env != NULL? //Path of onnxim(== TOGSim)

// After
_noc_node_per_core = config.icnt_injection_ports_per_core; // Number of NoC injection ports per core
char* onnxim_path_env = std::getenv("TORCHSIM_DIR"); // Path of onnxim(== TOGSim)
std::string onnxim_path = onnxim_path_env != NULL ? // Path of onnxim(== TOGSim)
```

---

## Files Modified
| File | Change |
|---|---|
| `configs/ramulator2_configs/ddr5_4800an_16ch.yaml` | Frontend: MessReqGenerator → GEM5 |
| `TOGSim/extern/ramulator2/src/dram/impl/DDR5.cpp` | Added DDR5_4800AN timing preset + extended rate lookup tables |
| `TOGSim/src/Simulator.cc` | Fixed missing `//` comment markers |

## Rebuild Required After Changes
```bash
cd /workspace/PyTorchSim/TOGSim/build && make -j$(nproc)
```

## Environment Variables Required
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export TORCHSIM_DIR=/workspace/PyTorchSim
export TOGSIM_CONFIG=/workspace/PyTorchSim/configs/samsung_mx_npu.yml
export PATH=/usr/local/cuda-12.1/bin:$PATH
```
