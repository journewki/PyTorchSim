# Depthwise Conv2D MLIR template for grouped convolutions where groups == I_C == O_C.
# Each output channel depends only on the corresponding input channel (I_C_per_group=1).
# The outer loop iterates one group at a time (TILE_N=1) to avoid weight buffer overflow.

from sympy import Symbol, Number
from typing import List, Optional

from PyTorchSimFrontend.mlir.mlir_conv_common import MLIRConvCommonTemplate
from PyTorchSimFrontend.mlir.mlir_conv_sb_template import MLIRConvSingleBatchTemplate
from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplateKernel
from torch._inductor.ir import IRNode
from PyTorchSimFrontend.mlir import mlir_common

# Same as CONV_TEMPLATE in mlir_conv_sb_template.py except:
#   - tile_k loop upper bound is I_C_PER_GROUP instead of I_C
CONV_DEPTHWISE_TEMPLATE = r"""
// Depthwise Conv2D kernel (groups == I_C == O_C)
// BATCH = {{ BATCH }}
// G = {{ G }}
// I_C_PER_GROUP = {{ I_C_PER_GROUP }}
// I_H = {{ I_H }}
// I_W = {{ I_W }}
// O_C = {{ O_C }}
// K_H = {{ K_H }}
// K_W = {{ K_W }}
// O_H = {{ O_H }}
// O_W = {{ O_W }}
// TILE_M = {{ TILE_M }}
// TILE_N = {{ TILE_N }}
// TILE_K = {{ TILE_K }}
// TILE_I_H={{ TILE_I_H }},
// TILE_I_W={{ TILE_I_W }},
// TILE_O_H={{ TILE_O_H }},
// TILE_O_W={{ TILE_O_W }},
// TILE_K_H={{ TILE_K_H }},
// TILE_K_W={{ TILE_K_W }},
// SUB_TILE_M={{ SUB_TILE_M }},
// SUB_TILE_N={{ SUB_TILE_N }},
// SUB_TILE_I_W={{ SUB_TILE_I_W }},
// SUB_TILE_K_H={{ SUB_TILE_K_H }},
// SUB_TILE_K_W={{ SUB_TILE_K_W }},
// PADDING_H = {{ PADDING_H }}
// PADDING_W = {{ PADDING_W }}
// STRIDE_H = {{ STRIDE_H }}
// STRIDE_W = {{ STRIDE_W }}
// DATA_STYPE = {{ DATA_STYPE }}

#map_I_H = affine_map<(d0, d1) -> (d0 * {{ STRIDE_H }} + d1)>
#map_I_W = affine_map<(d0, d1) -> (d0 * {{ STRIDE_W }} + d1)>
#offset_w_map = affine_map<(d0, d1) -> (d0 * {{ TILE_K_W * TILE_K * TILE_N }} + d1 * {{ TILE_K * TILE_N }})>
#offset_x_map = affine_map<(d0, d1) -> (d0 * {{ kernel.get_spad_size_per_lane(TILE_I_W, TILE_K) }} + d1)>
#offset_y_map = affine_map<(d0, d1) -> (d0 * {{ kernel.get_spad_size_per_lane(TILE_M, TILE_N) }} + d1 * {{ kernel.get_spad_size_per_lane(TILE_M, TILE_N) }})>
{{kernel.def_global_vars()}}

func.func @{{ KERNEL_NAME }}{{kernel.def_conv_kernel(inputs=[X, W, BIAS], outputs=[Y], names_str="X, W, Bias, Y", padded_input_size=PADDED_INPUT_SIZE, input_reorder=input_reorder)}} {
  {{ kernel.def_sram_buffer("X", X_tile_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("W", W_tile_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("Y", Y_tile_desc, indent_size=2) }}
  %v0 = arith.constant dense<0.0> : vector<{{ kernel.get_spad_size_per_lane(TILE_O_H * TILE_M, TILE_N) }}xf32>
  %c0 = arith.constant 0 : index
  {{- kernel.def_local_vars(indent_size=2) }}
  affine.for %tile_n = 0 to {{ O_C }} step {{ TILE_N }} {
    affine.for %o_h = 0 to {{ O_H }} step {{ TILE_O_H }} {
      affine.for %tile_m = 0 to {{ O_W }} step {{ TILE_M }} {
        // Initialize output
        {%- if BIAS %}
        {{ kernel.def_dma_op("MVIN", "Bias", Bias_idx, Bias_tile_desc, subtile_size=[1, SUB_TILE_N, TILE_O_H, SUB_TILE_M], indent_size=8) }}
        {%- else %}
        affine.vector_store %v0, %output_buffer[%c0, %c0, %c0, %c0] : {{ Y_tile_desc.get_mlir_shape(DATA_STYPE) }}, vector<{{ kernel.get_spad_size_per_lane(TILE_O_H * TILE_M, TILE_N) }}xf32>
        {%- endif %}
        affine.for %k_h = 0 to {{ K_H }} step {{ TILE_K_H }} {
          affine.for %k_w = 0 to {{ K_W }} step {{ TILE_K_W }} {
            affine.for %tile_k = 0 to {{ I_C_PER_GROUP }} step {{ TILE_K }} {
              %index_i_h = affine.apply #map_I_H(%o_h, %k_h)
              %index_i_w = affine.apply #map_I_W(%tile_m, %k_w)
              // Load input & weight matrix
              {{ kernel.def_dma_op("MVIN", "X", X_idx, X_tile_desc, subtile_size=[1, SUB_TILE_I_H, SUB_TILE_M, SUB_TILE_K], indent_size=14) }}
              {{ kernel.def_dma_op("MVIN", "W", W_idx, W_tile_desc, subtile_size=[SUB_TILE_K_H, SUB_TILE_K_W, SUB_TILE_K, SUB_TILE_N], indent_size=14) }}
              // Compute body part
              affine.for %tile_k_h = 0 to {{ TILE_K_H }} { // loop order should be fixed for timing simulation. Do not change this order.
                affine.for %tile_k_w = 0 to {{ TILE_K_W }} {
                  %offset_w = affine.apply #offset_w_map(%tile_k_h, %tile_k_w)
                  %W_buffer = memref.reinterpret_cast %weight_buffer to offset: [%offset_w], sizes: [{{ TILE_K }}, {{ TILE_N }}], strides: [{{ TILE_N }}, 1] : {{ W_tile_desc.get_mlir_shape(DATA_STYPE) }} to memref<{{ TILE_K }}x{{ TILE_N }}xf32, strided<[{{ TILE_N }}, 1], offset: ?>, 1>
                  affine.for %tile_o_h = 0 to {{ TILE_O_H }} {
                    affine.for %tile_o_w = 0 to {{ 1 }} { // TILE_O_W
                      %tile_i_h = affine.apply #map_I_H(%tile_o_h, %tile_k_h)
                      %offset_x = affine.apply #offset_x_map(%tile_i_h, %tile_k_w)
                      %offset_y = affine.apply #offset_y_map(%tile_o_h, %tile_o_w)
                      %X_buffer = memref.reinterpret_cast %input_buffer to offset: [%offset_x], sizes: [{{ TILE_M }}, {{ TILE_K }}], strides: [{{ TILE_K }}, 1] : {{ X_tile_desc.get_mlir_shape(DATA_STYPE) }} to memref<{{ TILE_M }}x{{ TILE_K }}xf32, strided<[{{ TILE_K }}, 1], offset: ?>, 1>
                      %Y_buffer = memref.reinterpret_cast %output_buffer to offset: [%offset_y], sizes: [{{ TILE_M }}, {{ TILE_N }}], strides: [{{ TILE_N }}, 1] : {{ Y_tile_desc.get_mlir_shape(DATA_STYPE) }} to memref<{{ TILE_M }}x{{ TILE_N }}xf32, strided<[{{ TILE_N }}, 1], offset: ?>, 1>
                      linalg.matmul ins(%X_buffer, %W_buffer : memref<{{ TILE_M }}x{{ TILE_K }}xf32, strided<[{{ TILE_K }}, 1], offset: ?>, 1>, memref<{{ TILE_K }}x{{ TILE_N }}xf32, strided<[{{ TILE_N }}, 1], offset: ?>, 1>)
                            outs(%Y_buffer : memref<{{ TILE_M }}x{{ TILE_N }}xf32, strided<[{{ TILE_N }}, 1], offset: ?>, 1>)
                    } { inner_loop=true }
                  } { inner_loop=true }
                } { inner_loop=true }
              } { inner_loop=true }
            } { accumulation_loop=true, subtile_loop="k" }
          } { accumulation_loop=true }
        } { accumulation_loop=true }
        // Store output matrix
        {{kernel.store_output(indent_size=8)}}
      } { outer_loop=true, subtile_loop="m" }
    } { outer_loop=true }
  } { outer_loop=true, subtile_loop="n" }
  return
}
"""


class MLIRConvDepthwiseTemplate(MLIRConvCommonTemplate):
    # Reuse the same Python wrapper as MLIRConvSingleBatchTemplate — padding, permuting
    # X→NHWC and W→(K_H, K_W, I_C_per_group, O_C) are identical operations.
    WRAPPER_TEMPLATE = MLIRConvSingleBatchTemplate.WRAPPER_TEMPLATE

    def __init__(self, input_nodes, layout, input_reorder=None, **kwargs):
        super().__init__(input_nodes, layout, input_reorder, **kwargs)
        self.groups = kwargs["groups"]
        # Prefix function name to avoid cache collision with non-depthwise conv of same shapes
        self.function_name = "DepthwiseConv2D_" + "_".join(self.input_shape) \
            + "_".join(self.weight_shape) \
            + "_" + "_".join([str(i) for i in self.stride]) \
            + "_" + "_".join([str(i) for i in self.padding]) \
            + "_" + "_".join([str(i) for i in self.dilation])

    def render(self,
               kernel: MLIRTemplateKernel,
               template_buffer_node=None,
               epilogue_nodes: Optional[List[IRNode]] = None,
               tile_info=None,
               **kwargs):
        X, W, Y, Bias, n_extra_node, BATCH, I_C, I_H, I_W, O_C, K_H, K_W, O_H, O_W, PADDING_H, PADDING_W, STRIDE_H, STRIDE_W = self.extract_info(kernel, template_buffer_node, epilogue_nodes)

        G = self.groups
        I_C_per_group = int(I_C) // G

        if tile_info is None:
            TILE_K_H, TILE_K_W, TILE_O_H, TILE_O_W, TILE_M, TILE_N, TILE_K, TILE_I_H, TILE_I_W, SUB_TILE_I_H, SUB_TILE_I_W, SUB_TILE_K_H, SUB_TILE_K_W, SUB_TILE_M, SUB_TILE_N, SUB_TILE_K = self.select_tile(kernel, n_extra_node, BATCH, I_C, O_C, K_H, K_W, O_H, O_W)[0]
        else:
            TILE_K_H, TILE_K_W, TILE_O_H, TILE_O_W, TILE_M, TILE_N, TILE_K, TILE_I_H, TILE_I_W, SUB_TILE_I_H, SUB_TILE_I_W, SUB_TILE_K_H, SUB_TILE_K_W, SUB_TILE_M, SUB_TILE_N, SUB_TILE_K = tile_info

        SUB_TILE_N = TILE_N if TILE_N > 512 else SUB_TILE_N
        TOG_latency = O_W if TILE_M > O_W else TILE_M
        TOG_latency = 8 if TOG_latency < 8 else TOG_latency
        kernel.loop_size = [TOG_latency, TILE_N, TILE_K]

        vlane_stride = 1
        vlane_split_axis = 1

        X_tile_size = [1, TILE_I_H, TILE_I_W, TILE_K]
        X_tile_stride = [TILE_I_H * TILE_I_W * TILE_K, TILE_I_W * TILE_K, 1, TILE_I_W]
        X_tile_desc = mlir_common.MLIRMultiDimTile(X_tile_size, kernel.vector_lane, 3, vlane_stride)
        X_tile_desc.set_tile_size_stride(X_tile_size, X_tile_stride)
        X_tile_desc.set_name("input_buffer")
        # Key difference: channel index for depthwise is tile_n (the group), not tile_k
        X_dim = [Symbol("c0"), Symbol("index_i_h"), Symbol("index_i_w"), Symbol("tile_n")]
        X_idx = [X_dim[0] * ((I_W + 2 * PADDING_W) * (I_H + 2 * PADDING_H) * I_C),
                 X_dim[1] * ((I_W + 2 * PADDING_W) * I_C),
                 X_dim[2] * I_C,
                 X_dim[3]]

        W_tile_size = [TILE_K_H, TILE_K_W, TILE_K, TILE_N]
        W_tile_stride = [TILE_K_W * TILE_K * TILE_N, TILE_K * TILE_N, 1, TILE_K]
        W_tile_desc = mlir_common.MLIRMultiDimTile(X_tile_size, kernel.vector_lane, 3, vlane_stride)
        W_tile_desc.set_tile_size_stride(W_tile_size, W_tile_stride)
        W_tile_desc.set_name("weight_buffer")
        # Key difference: W strides use I_C_per_group instead of I_C
        W_dim = [Symbol("k_h"), Symbol("k_w"), Symbol("tile_k"), Symbol("tile_n")]
        W_idx = [W_dim[0] * K_W * I_C_per_group * O_C,
                 W_dim[1] * I_C_per_group * O_C,
                 W_dim[2] * O_C,
                 W_dim[3]]

        Y_tile_size = [1, TILE_N, TILE_O_H, TILE_M]
        Y_tile_stride = [TILE_O_H * TILE_M * TILE_N, TILE_M, TILE_M * TILE_N, 1]
        Y_tile_desc = mlir_common.MLIRMultiDimTile(Y_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        Y_tile_desc.set_tile_size_stride(Y_tile_size, Y_tile_stride)
        Y_tile_desc.set_name("output_buffer")
        Y_idx = [Number(0), Symbol("tile_n") * O_H * O_W, Symbol("o_h") * O_W, Symbol("tile_m")]

        Bias_idx = [Number(0), Symbol("tile_n"), Number(0), Number(0)]
        Bias_tile_desc = mlir_common.MLIRMultiDimTile(Y_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        Bias_tile_desc.set_tile_size_stride(Y_tile_size, Y_tile_stride)
        Bias_tile_desc.set_name("output_buffer")
        if Bias is not None:
            Bias_tile_desc.offset = Bias.get_layout().offset

        kernel.render_options = dict(
            KERNEL_NAME=self.name,
            kernel=kernel,
            X=X, W=W, Y=Y, BIAS=Bias,
            PADDED_INPUT_SIZE=self.get_padded_input_size(X),
            BATCH=BATCH,
            G=G,
            I_C=I_C,
            I_C_PER_GROUP=I_C_per_group,
            I_H=I_H,
            I_W=I_W,
            O_C=O_C,
            K_H=K_H,
            K_W=K_W,
            O_H=O_H,
            O_W=O_W,
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            TILE_K=TILE_K,
            TILE_I_H=TILE_I_H,
            TILE_I_W=TILE_I_W,
            TILE_O_H=TILE_O_H,
            TILE_O_W=TILE_O_W,
            TILE_K_H=TILE_K_H,
            TILE_K_W=TILE_K_W,
            SUB_TILE_M=SUB_TILE_M,
            SUB_TILE_N=SUB_TILE_N,
            SUB_TILE_K=SUB_TILE_K,
            SUB_TILE_I_H=SUB_TILE_I_H,
            SUB_TILE_I_W=SUB_TILE_I_W,
            SUB_TILE_K_H=SUB_TILE_K_H,
            SUB_TILE_K_W=SUB_TILE_K_W,
            PADDING_H=PADDING_H,
            PADDING_W=PADDING_W,
            STRIDE_H=STRIDE_H,
            STRIDE_W=STRIDE_W,
            X_tile_desc=X_tile_desc,
            W_tile_desc=W_tile_desc,
            Y_tile_desc=Y_tile_desc,
            Bias_tile_desc=Bias_tile_desc,
            X_idx=X_idx,
            W_idx=W_idx,
            Bias_idx=Bias_idx,
            DATA_STYPE="f32",
            input_reorder=self.input_reorder,
        )

        kernel.epilogue_info = dict(
            output_node=self.output_node.name,
            sram_var="output_buffer",
            dram_var="Y",
            dram_idx=Y_idx,
            dram_tile_desc=Y_tile_desc,
            dim_aliasing={"index0": "c0", "index1": "tile_n", "index2": "o_h", "index3": "tile_m"},
        )
        kernel.exception_nodes["X"] = {"numel": (I_W + 2 * PADDING_W) * (I_H + 2 * PADDING_H) * I_C * BATCH}
        code = self._template_from_string(CONV_DEPTHWISE_TEMPLATE).render(**kernel.render_options)
        kernel.add_loop_info(
            [K_H, K_W, O_H, O_W, BATCH, G, I_C_per_group],
            [TILE_M, TILE_N, TILE_K],
        )
        return code

    def select_tile(self, kernel, n_extra_node, BATCH, I_C, O_C, K_H, K_W, O_H, O_W):
        G = self.groups
        I_C_per_group = int(I_C) // G

        # Each depthwise group uses a different input channel, so TILE_N must be 1:
        # the matmul X_buffer holds only 1 channel (I_C_per_group=1), and that channel
        # belongs exclusively to the current group (tile_n). Processing TILE_N>1 groups
        # simultaneously would apply the same input channel to all N groups, which is wrong.
        TILE_N = 1
        TILE_K = I_C_per_group  # = 1 for depthwise
        TILE_K_H = K_H
        TILE_K_W = K_W
        TILE_O_H = O_H
        TILE_O_W = O_W
        TILE_M = O_W

        TILE_I_H = 1 + (TILE_O_H - 1) * self.stride[0] + (TILE_K_H - 1) * self.dilation[0]
        TILE_I_W = 1 + (TILE_O_W - 1) * self.stride[1] + (TILE_K_W - 1) * self.dilation[1]

        SUB_TILE_I_H = 1
        SUB_TILE_I_W = 1
        SUB_TILE_K_H = 1
        SUB_TILE_K_W = 1
        SUB_TILE_M = TILE_I_W if TILE_I_W < kernel.vector_lane else kernel.vector_lane
        SUB_TILE_N = TILE_N
        SUB_TILE_K = TILE_K

        return [(TILE_K_H, TILE_K_W, TILE_O_H, TILE_O_W, TILE_M, TILE_N, TILE_K,
                 TILE_I_H, TILE_I_W,
                 SUB_TILE_I_H, SUB_TILE_I_W, SUB_TILE_K_H, SUB_TILE_K_W,
                 SUB_TILE_M, SUB_TILE_N, SUB_TILE_K)]
