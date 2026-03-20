#!/bin/bash
# =============================================================================
# SANA + FALQON PyTorchSim Test Runner
#
# Phase 1: conda diffusers + CUDA verification
# Phase 2: PyTorchSim NPU simulation (uncomment npu section)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/test_sana_falqon.py"

# --- Phase 1: CUDA verification (conda diffusers) ---
echo "============================================="
echo "Phase 1: CUDA Verification (conda diffusers)"
echo "============================================="

python "$SCRIPT" \
    --device cuda \
    --rank 4 \
    --dtype float32 \
    --rtol 1e-3 \
    --atol 1e-3 \
    --tests "all" \
    --num_steps 5

# --- Individual test examples ---
# python "$SCRIPT" --device cuda --tests "falqon_linear"
# python "$SCRIPT" --device cuda --tests "linear_attention,cross_attention"
# python "$SCRIPT" --device cuda --tests "transformer_block,falqon_block"
# python "$SCRIPT" --device cuda --tests "training_step"
# python "$SCRIPT" --device cuda --tests "flow_matching" --num_steps 10

# --- Phase 2: PyTorchSim NPU simulation (uncomment when environment is ready) ---
# echo "============================================="
# echo "Phase 2: PyTorchSim NPU Simulation"
# echo "============================================="
# python "$SCRIPT" \
#     --device npu \
#     --rank 4 \
#     --dtype float32 \
#     --rtol 1e-3 \
#     --atol 1e-3 \
#     --tests "all"