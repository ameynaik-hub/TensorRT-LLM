import math
import weakref
from typing import Optional, Union, cast

import torch

torch.set_printoptions(precision=3, sci_mode=True)
from torch import nn

# Import llti operations to register them with PyTorch
try:
    import llti.ops._C  # This registers the operations
    LLTI_AVAILABLE = True
except ImportError:
    LLTI_AVAILABLE = False
    print(
        "Warning: llti module not available, fused operations will be disabled")

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..attention_backend import (AttentionInputType, AttentionMetadata,
                                 TrtllmAttention, TrtllmAttentionMetadata)
from ..attention_backend.interface import (AttentionMask,
                                           PositionalEmbeddingParams,
                                           PredefinedAttentionMask)
from ..attention_backend.utils import create_attention, get_attention_backend
from ..distributed import AllReduceParams
from ..model_config import ModelConfig
from ..peft.lora.layer import LoraLayer, LoraModuleType
from ..utils import Fp4QuantizedTensor, get_model_extra_attrs
from .linear import Linear, TensorParallelMode, WeightMode, WeightsLoadingConfig
from .multi_stream_utils import maybe_execute_in_parallel, maybe_execute_in_parallel_3streams
from .rms_norm import RMSNorm
from .rotary_embedding import RotaryEmbedding


def swizzle_b_matrix(B, weight_block_slice_k_factor, weight_block_slice_k_cols):
    """
    Swizzles a matrix B into a 4D blocked layout.

    The original matrix B of shape (k, n) is conceptually transformed into a
    4D tensor of shape (k/F_row, n/F_col, F_row, F_col) and then
    flattened to a 1D array.

    The mapping is:
    Dest[block_row, block_col, row_in_block, col_in_block] = B[
        block_row + row_in_block * (k / F_row),
        block_col * F_col + col_in_block
    ]

    Args:
        B (torch.Tensor): The input matrix of shape (k, n).
        weight_block_slice_k_factor (int): F_row, the number of rows in a block.
        weight_block_slice_k_cols (int): F_col, the number of columns in a block.

    Returns:
        torch.Tensor: The swizzled matrix, flattened to a 2D array
                     of shape (k/F_row, n*F_row).
    """
    k, n = B.shape

    F_row = weight_block_slice_k_factor
    F_col = weight_block_slice_k_cols

    if k % F_row != 0:
        raise ValueError(
            "k must be divisible by weight_block_slice_k_factor (F_row)")
    if n % F_col != 0:
        raise ValueError(
            "n must be divisible by weight_block_slice_k_cols (F_col)")

    num_block_rows = k // F_row
    num_block_cols = n // F_col

    # Create index tensors for vectorized access
    i0 = torch.arange(num_block_rows, device=B.device)[:, None, None, None]
    i1 = torch.arange(num_block_cols, device=B.device)[None, :, None, None]
    i2 = torch.arange(F_row, device=B.device)[None, None, :, None]
    i3 = torch.arange(F_col, device=B.device)[None, None, None, :]

    # Calculate source indices vectorized
    src_row = i0 + i2 * num_block_rows
    src_col = i1 * F_col + i3

    # Gather from source indices in one operation
    B_swizzled_4d = B[src_row, src_col]

    # The C++ code stores this 4D tensor linearly, which is equivalent
    # to reshaping it to 2D for easier use.
    # The final physical layout is (k/F_row) x (n*F_row).
    return B_swizzled_4d.reshape(k, n)


class Attention(nn.Module):

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        max_position_embeddings: int,
        bias: bool,
        pos_embd_params: Optional[PositionalEmbeddingParams] = None,
        rope_fusion: Optional[bool] = None,
        layer_idx: Optional[int] = None,
        dtype: torch.dtype = None,
        dense_bias: Optional[bool] = None,
        config: Optional[ModelConfig] = None,
        q_scaling: float = 1.0,
        attention_chunk_size: Optional[int] = None,
    ):
        """
        Initialize the Attention module.

        Args:
            hidden_size (int): The size of the hidden dimension.
            num_attention_heads (int): The number of attention heads.
            num_key_value_heads (int): The number of key value heads.
            max_position_embeddings (int): The maximum position embeddings.
            bias (bool): Whether to use bias in the linear layers.
            pos_embd_params (Optional[PositionalEmbeddingParams]): The positional embedding parameters.
            rope_fusion (Optional[bool]): Whether to fuse RoPE into the attention OP and skip applying unfused RoPE. If None, whether to fuse is decided by the capability of the attention backend.
            layer_idx (Optional[int]): The layer index.
            dtype (torch.dtype): The data type.
            dense_bias (Optional[bool]): Whether to use bias in the output projection layer.
            config (Optional[ModelConfig]): The model configuration.
            q_scaling (float): The scaling factor for the qk_scale. The definition is $O = softmax(QK^T * qk_scale) * V, qk_scale = 1 / (sqrt(head_dim) * q_scaling)$. The default value is 1.0.
            attention_chunk_size (Optional[int]): See [Chunked Attention] below.
        """
        super().__init__()
        self.layer_idx = layer_idx

        config = config or ModelConfig()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = getattr(config.pretrained_config, 'head_dim', None)
        if not isinstance(self.head_dim, int):
            self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.pos_embd_params = pos_embd_params
        self.dense_bias = dense_bias
        self.q_scaling = q_scaling

        # [Chunked Attention]
        # Chunked attention is applied to context requests only. Chunked attention will be
        # applied when this field is specified and mMaskType == CAUSAL.
        #
        # In chunked attention, we break context requests into chunks of a specified size. Tokens can only
        # attend to tokens in the same chunk. So, for example, if the chunk size is 3, we might have a mask
        # that looks like this:
        #
        # 1 0 0 0 0 0
        # 1 1 0 0 0 0
        # 1 1 1 0 0 0
        # 0 0 0 1 0 0
        # 0 0 0 1 1 0
        # 0 0 0 1 1 1
        self.attention_chunk_size = attention_chunk_size

        if dense_bias is None:
            self.dense_bias = bias

        # tensor parallel
        tp_size = config.mapping.tp_size
        pp_size = config.mapping.pp_size
        if config.mapping.enable_attention_dp:
            tp_size = 1

        mapping = Mapping(
            world_size=tp_size * pp_size,
            tp_size=tp_size,
            pp_size=pp_size,
            rank=config.mapping.rank,
            gpus_per_node=config.mapping.gpus_per_node,
            enable_attention_dp=config.mapping.enable_attention_dp,
        )
        assert self.num_heads % tp_size == 0
        self.num_heads = self.num_heads // tp_size
        self.num_key_value_heads = (self.num_key_value_heads + tp_size -
                                    1) // tp_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim

        self.qkv_proj = Linear(
            self.hidden_size,
            tp_size * self.q_size + 2 * tp_size * self.kv_size,
            bias=bias,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_QKV_LINEAR),
            quant_config=config.get_quant_config(),
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            allreduce_strategy=config.allreduce_strategy,
            force_dynamic_quantization=config.force_dynamic_quantization)
        self.o_lora = LoraLayer([LoraModuleType.ATTENTION_DENSE],
                                [self.hidden_size])

        self.o_proj = Linear(
            tp_size * self.q_size,
            self.hidden_size,
            bias=self.dense_bias,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            quant_config=config.get_quant_config(),
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            lora=self.o_lora,
            allreduce_strategy=config.allreduce_strategy,
            force_dynamic_quantization=config.force_dynamic_quantization)

        self.quant_config = config.get_quant_config()
        self.attn_backend = config.attn_backend
        attn_cls = get_attention_backend(self.attn_backend)

        # These two modules are mutually exclusive - either splitted_qkv_lora or fused_qkv_lora will be used,
        # but never both at the same time. splitted_qkv_lora handles Q,K,V separately while fused_qkv_lora
        # handles them as a single fused operation.
        self.splitted_qkv_lora = LoraLayer([
            LoraModuleType.ATTENTION_Q, LoraModuleType.ATTENTION_K,
            LoraModuleType.ATTENTION_V
        ], [self.q_size, self.kv_size, self.kv_size])
        self.fused_qkv_lora = LoraLayer([LoraModuleType.ATTENTION_QKV],
                                        [self.q_size + 2 * self.kv_size])

        self.o_lora = LoraLayer([LoraModuleType.ATTENTION_DENSE],
                                [self.hidden_size])

        # Whether to fuse RoPE into the attention OP.
        # If true, RoPE will be applied in self.attn.forward.
        # If false, RoPE will be applied in self.apply_rope.
        self.rope_fusion = rope_fusion
        if self.rope_fusion and not attn_cls.support_fused_rope():
            logger.warning(
                "rope_fusion is true but the attention backend does not support it. Will disable rope_fusion."
            )
            self.rope_fusion = False
        # If rope_fusion is not specified, enable if the attention backend supports it.
        if self.rope_fusion is None:
            self.rope_fusion = attn_cls.support_fused_rope()

        self.rotary_emb = None
        if not self.rope_fusion and self.pos_embd_params is not None:
            self.rotary_emb = RotaryEmbedding(
                self.pos_embd_params.rope,
                head_dim=self.head_dim,
                is_neox=self.pos_embd_params.is_neox,
            )

        self.attn = create_attention(
            self.attn_backend,
            self.layer_idx,
            self.num_heads,
            self.head_dim,
            self.num_key_value_heads,
            pos_embd_params=self.pos_embd_params if self.rope_fusion else None,
            quant_config=self.quant_config,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            q_scaling=self.q_scaling,
            attention_chunk_size=self.attention_chunk_size,
        )

        self.support_fused_qkv = self.attn.support_fused_qkv()
        self.support_nvfp4_output = self.attn.support_nvfp4_output()

        if not config.skip_create_weights_in_init:
            self.create_weights()

    def create_weights(self):
        # self.attn has no weights but has states that are related to quant_config,
        # which could be modified after __init__
        self.attn.update_quant_config(self.quant_config)

    def split_qkv(self, q, k=None, v=None):
        if k is None and v is None:
            q, k, v = q.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return q, k, v

    def convert_qkv(self, q, k, v):
        if k is None and v is None and not self.support_fused_qkv:
            q, k, v = self.split_qkv(q)
        elif k is not None and v is not None and self.support_fused_qkv:
            qkv = torch.concat([q, k, v], dim=-1)
            q, k, v = qkv, None, None
        return q, k, v

    def forward(
        self,
        position_ids: Optional[torch.IntTensor],
        hidden_states: Union[torch.Tensor, Fp4QuantizedTensor],
        attn_metadata: AttentionMetadata,
        attention_mask: AttentionMask = PredefinedAttentionMask.CAUSAL,
        mrope_config: Optional[dict] = None,
        all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
        attention_window_size: Optional[int] = None,
        attention_mask_data: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for the Attention module.

        Args:
            position_ids (Optional[torch.IntTensor]): The position IDs.
            hidden_states (torch.Tensor): The hidden states.
            attn_metadata (AttentionMetadata): The attention metadata.
            attention_mask (AttentionMask): The attention mask type.
            mrope_config (Optional[dict]): The MROPE configuration.
            all_reduce_params (Optional[AllReduceParams]): The all reduce parameters.
            lora_params (Optional[dict]): The LoRA parameters.
            attention_window_size (Optional[int]): The attention window size.
            attention_mask_data (Optional[torch.Tensor]): The attention mask data.
        Returns:
            torch.Tensor: The output tensor.
        """
        qkv = self.qkv_proj(hidden_states)

        if bool(lora_params):
            qkv_lora = self.splitted_qkv_lora(hidden_states, lora_params,
                                              self.layer_idx)
            if qkv_lora is not None:
                qkv = qkv + qkv_lora

            qkv_lora = self.fused_qkv_lora(hidden_states, lora_params,
                                           self.layer_idx)
            if qkv_lora is not None:
                qkv = qkv + qkv_lora

        q, k, v = qkv, None, None

        q, k, v = self.apply_rope(q, k, v, position_ids)

        out_scale = None
        out_scale_sf = None
        if self.o_proj.has_fp8_qdq or self.o_proj.has_nvfp4 or self.o_proj.has_fp8_block_scales or self.o_proj.has_fp8_rowwise:
            out_scale = self.o_proj.inv_input_scale
        if self.o_proj.has_nvfp4 and self.support_nvfp4_output:
            out_scale_sf = self.o_proj.input_scale

        q, k, v = self.convert_qkv(q, k, v)
        attn_output = self.attn.forward(
            q,
            k,
            v,
            attn_metadata,
            out_scale=out_scale,
            out_scale_sf=out_scale_sf,
            attention_mask=attention_mask,
            mrope_config=mrope_config,
            attention_window_size=attention_window_size,
            attention_mask_data=attention_mask_data)
        hidden_states = attn_output
        attn_output = self.o_proj(attn_output,
                                  all_reduce_params=all_reduce_params,
                                  lora_params=lora_params,
                                  layer_idx=self.layer_idx)
        return attn_output

    def apply_rope(self, q: torch.Tensor, k: Optional[torch.Tensor],
                   v: Optional[torch.Tensor], position_ids: torch.Tensor):
        """
        Apply RoPE to the query and key.
        Depending on the implementation, q, k, v could be either fused (q, k, v = concat(q, k, v), None, None) or unfused (none of q, k, v is None).
        Before self.attn.forward, convert_qkv will be called to make sure that the format of (q, k, v) satisfies the requirement of self.attn.
        This method could be overridden in the subclass, in which extra functionalities such as q_norm/k_norm could be added.
        Args:
            q (torch.Tensor): The query tensor.
            k (Optional[torch.Tensor]): The key tensor.
            v (Optional[torch.Tensor]): The value tensor.
            position_ids (torch.Tensor): The position IDs of each token for RoPE.
        Returns:
            tuple: A tuple of (q, k, v).
        """
        q, k, v = self.split_qkv(q, k, v)
        # If RoPE is fused into the attention OP, do not apply RoPE here.
        if not self.rope_fusion and position_ids is not None:
            q, k = self.rotary_emb(position_ids, [q, k])
        return q, k, v


def extract_extra_attrs(layer_idx: str):
    extra_attrs = get_model_extra_attrs()
    assert extra_attrs is not None, "Model extra attrs is not set"

    metadata_ref = extra_attrs.get("attention_metadata", None)
    assert metadata_ref is not None, "Attention metadata is not set"
    metadata = metadata_ref()
    assert isinstance(
        metadata,
        TrtllmAttentionMetadata,
    )

    mla_layers = extra_attrs.get("mla_layers", None)
    assert mla_layers is not None, "MLA layers is not registered"
    mla_layer_ref = mla_layers.get(layer_idx, None)
    assert mla_layer_ref is not None, f"Cannot find MLA layer for layer {layer_idx}"
    mla_layer = mla_layer_ref()
    assert isinstance(
        mla_layer,
        MLA), "MLA layer must be a subclass of MLA or an instance of MLA"

    return metadata, mla_layer


@torch.library.custom_op("trtllm::mla_custom_op_inplace",
                         mutates_args=("output", ))
def mla_custom_op_inplace(
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
    output: torch.Tensor,
) -> None:
    metadata, mla_layer = extract_extra_attrs(layer_idx)
    mla_layer.forward_impl(position_ids, hidden_states, metadata, output=output)


def fp8_block_scaling_bmm_out(
    mat1: torch.Tensor,
    mat2_fp8: torch.Tensor,
    mat2_scale: torch.Tensor,
    out: torch.Tensor,
    mat2_dequant: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    sm_version = get_sm_version()
    if sm_version == 90 or sm_version == 89:
        mat1_fp8, mat1_scale = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(
            mat1)
        torch.ops.trtllm.fp8_block_scaling_bmm_out(mat1_fp8, mat2_fp8,
                                                   mat1_scale, mat2_scale, out)
    elif sm_version == 100:
        output = torch.bmm(mat1.transpose(0, 1), mat2_dequant.transpose(1, 2))
        out.copy_(output)

        # low_latency = True
        # use_deep_seek_fp8 = True
        # tile_size = 8
        # epilogue_tile_m = 64 if use_deep_seek_fp8 else 128
        # m_size = mat1.shape[0]
        # if m_size % tile_size != 0:
        #     tiled_shape = ((m_size + tile_size - 1) // tile_size) * tile_size
        #     mat1 = torch.nn.functional.pad(
        #         mat1, (0, 0, 0, 0, 0, tiled_shape - m_size), "constant", 0)

        # mat1_fp8, mat1_scale = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(
        #     mat1)
        # output, output_sf = torch.ops.trtllm.fp8_batched_gemm_trtllmgen(
        #     mat1_fp8,
        #     mat2_fp8,
        #     tile_size=tile_size,
        #     epilogue_tile_m=epilogue_tile_m,
        #     use_deep_seek_fp8=use_deep_seek_fp8,
        #     low_latency=low_latency,
        #     dq_sfs_a=mat1_scale.reshape(mat1.shape[-1] // 128, -1),
        #     dq_sfs_b=mat2_scale,
        #     out_dtype=out.dtype,
        # )
        # out.copy_(output[:, :m_size])
    else:
        raise NotImplementedError(f"SM{sm_version} is not supported")


class MLA(nn.Module):

    def __init__(
            self,
            *,
            hidden_size: int,
            num_attention_heads: int,
            num_key_value_heads: int,
            qk_nope_head_dim: int,
            qk_rope_head_dim: int,
            v_head_dim: int,
            q_lora_rank: int,
            kv_lora_rank: int,
            predicted_tokens_per_seq: int,
            max_position_embeddings: int,
            bias: bool,
            aux_stream: Optional[torch.cuda.Stream] = None,
            concat_stream: Optional[torch.cuda.Stream] = None,
            pos_embd_params: Optional[PositionalEmbeddingParams] = None,
            layer_idx: Optional[int] = None,
            dtype: torch.dtype = None,
            dense_bias: Optional[bool] = None,
            config: Optional[ModelConfig] = None,
            split_q_gemm: bool = True,
            use_fused_bmm_for_generation:
        bool = True,  # Use BMM output from tk_bmm_fused_matmul for forward_generation
    ):
        """
        Initialize the MLA module.

        Args:
            hidden_size (int): The size of the hidden dimension.
            num_attention_heads (int): The number of attention heads.
            num_key_value_heads (int): The number of key value heads.
            qk_nope_head_dim (int): The dimension of the query and key without Rope.
            qk_rope_head_dim (int): The dimension of the Rope of query and key.
            v_head_dim (int): The dimension of the value.
            q_lora_rank (int): The dimension of the compressed query.
            kv_lora_rank (int): The dimension of the compressed key and value.
            predicted_tokens_per_seq (int): The number of predicted tokens per sequence.
            max_position_embeddings (int): The maximum position embeddings.
            bias (bool): Whether to use bias in the linear layers.
            aux_stream (Optional[torch.cuda.Stream]): The auxiliary CUDA stream for running operations in two parallel streams.
            concat_stream (Optional[torch.cuda.Stream]): The CUDA stream specifically for concat operations to overlap with computation.
            pos_embd_params (PositionalEmbeddingParams): The positional embedding parameters.
            layer_idx (int): The layer index.
            dtype (torch.dtype): The data type.
            dense_bias (bool): Whether to use bias in the output projection layer.
            config (ModelConfig): The model configuration.
            split_q_gemm (bool): Whether to split q_b_proj into separate nope and rope gemms. Default is False.
            use_fused_bmm_for_generation (bool): Use BMM output from tk_bmm_fused_matmul for forward_generation BMM. Default is True.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_idx_str = str(layer_idx)
        self.dtype = dtype

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.predicted_tokens_per_seq = predicted_tokens_per_seq
        self.max_position_embeddings = max_position_embeddings
        self.pos_embd_params = pos_embd_params
        self.dense_bias = dense_bias
        self.split_q_gemm = split_q_gemm
        self.use_fused_bmm_for_generation = use_fused_bmm_for_generation
        if dense_bias is None:
            self.dense_bias = bias

        # Initialize split weight storage for split_q_gemm
        if self.split_q_gemm:
            self._split_weights_initialized = False
            self._q_nope_weight = None
            self._q_rope_weight = None
            self._q_nope_bias = None
            self._q_rope_bias = None
            # Additional storage for swizzled weights for tk_bmm_fused_matmul
            self._q_nope_weight_swizzled = None
            self._q_mixed_nope_weight = None  # Optimized weight that directly produces concat->view->split result
            self._q_mixed_rope_weight = None  # Optimized weight that directly produces rope concat->view->split result
            self._bmm_weights = None  # BMM weights for the fused operation (k_b_proj_trans when available)
            self._bmm_output_buffer = None  # Output buffer for BMM result

            # Storage for BMM output from tk_bmm_fused_matmul to use in forward_generation
            # self._cached_bmm_output = None  # BMM output from fused operation for reuse
            self._bmm_weights_updated = False  # Track if BMM weights have been updated

        if self.q_lora_rank is None:
            self.q_lora_rank = hidden_size
            self.is_lite = True
        else:
            self.is_lite = False

        assert pos_embd_params is not None, "pos_embd_params must be provided in MLA"

        self.register_to_config = False
        if config is not None:
            if "mla_layers" not in config.extra_attrs:
                config.extra_attrs["mla_layers"] = {}
            config.extra_attrs["mla_layers"][self.layer_idx_str] = weakref.ref(
                self)
            self.register_to_config = True

        # tensor parallel
        config = config or ModelConfig()
        self.config = config  # Store config for tp_rank access in debug prints
        tp_size = config.mapping.tp_size
        pp_size = config.mapping.pp_size
        if config.mapping.enable_attention_dp:
            tp_size = 1

        mapping = Mapping(
            world_size=tp_size * pp_size,
            tp_size=tp_size,
            pp_size=pp_size,
            rank=config.mapping.rank,
            gpus_per_node=config.mapping.gpus_per_node,
            enable_attention_dp=config.mapping.enable_attention_dp,
        )

        assert self.num_heads % tp_size == 0
        self.num_heads = self.num_heads // tp_size
        self.num_key_value_heads = (self.num_key_value_heads + tp_size -
                                    1) // tp_size

        rms_norm_eps = config.pretrained_config.rms_norm_eps
        quant_config = config.get_quant_config()
        self.quant_config = quant_config

        if not self.is_lite:
            self.kv_a_proj_with_mqa = Linear(
                hidden_size,
                self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                bias=bias,
                dtype=dtype,
                quant_config=quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                use_custom_cublas_mm=True,
                force_dynamic_quantization=config.force_dynamic_quantization)

            self.q_a_layernorm = RMSNorm(hidden_size=self.q_lora_rank,
                                         eps=rms_norm_eps,
                                         dtype=dtype)

            # Always create the original q_b_proj for weight storage
            self.q_b_proj = Linear(
                self.q_lora_rank,
                tp_size * self.num_heads * self.qk_head_dim,
                bias=bias,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                quant_config=quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                allreduce_strategy=config.allreduce_strategy,
                force_dynamic_quantization=config.force_dynamic_quantization)

            # For split_q_gemm, we'll split weights dynamically without separate Linear instances
        else:
            self.kv_a_proj_with_mqa = Linear(
                hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=bias,
                dtype=dtype,
                quant_config=quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                use_custom_cublas_mm=True,
                force_dynamic_quantization=config.force_dynamic_quantization)

            # Always create the original q_proj for weight storage (lite case)
            self.q_proj = Linear(
                self.q_lora_rank,
                tp_size * self.num_heads * self.qk_head_dim,
                bias=bias,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                quant_config=quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                allreduce_strategy=config.allreduce_strategy,
                force_dynamic_quantization=config.force_dynamic_quantization)
            self.q_b_proj = self.q_proj

        self.kv_a_layernorm = RMSNorm(hidden_size=kv_lora_rank,
                                      dtype=dtype,
                                      eps=rms_norm_eps)

        self.kv_b_proj = Linear(
            self.kv_lora_rank,
            tp_size * self.num_heads *
            (self.qk_nope_head_dim + self.v_head_dim),
            bias=bias,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=quant_config,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            allreduce_strategy=config.allreduce_strategy,
            force_dynamic_quantization=config.force_dynamic_quantization)
        # This parameter will view into self.kv_b_proj.weight after loading weights.
        # For dummy weight initialization, this parameter is initialized with empty tensor.
        # Used in forward_generation only
        self.v_b_proj = nn.Parameter(
            torch.empty(
                (self.num_heads, self.v_head_dim, self.kv_lora_rank),
                dtype=dtype,
            ),
            requires_grad=False,
        )

        self.o_proj = Linear(
            self.num_key_value_heads * self.v_head_dim * tp_size,
            self.hidden_size,
            bias=self.dense_bias,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            quant_config=quant_config,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            allreduce_strategy=config.allreduce_strategy,
            force_dynamic_quantization=config.force_dynamic_quantization)

        def yarn_get_mscale(scale=1, mscale=1):
            if scale <= 1:
                return 1.0
            return 0.1 * mscale * math.log(scale) + 1.0

        mscale_all_dim = pos_embd_params.rope.mscale_all_dim
        scaling_factor = pos_embd_params.rope.scale
        mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
        q_scaling = 1.0 / (mscale * mscale)

        self.mha = create_attention(
            config.attn_backend,
            self.layer_idx,
            self.num_heads,
            head_dim=self.qk_head_dim,
            num_kv_heads=self.num_key_value_heads,
            pos_embd_params=pos_embd_params,
            quant_config=quant_config,
            q_scaling=q_scaling,
            is_mla_enable=True,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            predicted_tokens_per_seq=self.predicted_tokens_per_seq,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
        )

        self.mqa = create_attention(
            config.attn_backend,
            self.layer_idx,
            self.num_heads,
            head_dim=self.kv_lora_rank + self.qk_rope_head_dim,
            num_kv_heads=1,
            pos_embd_params=pos_embd_params,
            quant_config=quant_config,
            q_scaling=q_scaling,
            is_mla_enable=True,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.kv_lora_rank,
            predicted_tokens_per_seq=self.predicted_tokens_per_seq,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
        )

        self.aux_stream = aux_stream
        self.concat_stream = concat_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event(), torch.cuda.Event()]  # Added 3rd event for concat

        self.rope_fusion = self.mha.support_fused_rope()
        self.support_fused_qkv = self.mha.support_fused_qkv()
        self.rotary_emb = None
        self.apply_rotary_emb = not self.rope_fusion
        if self.apply_rotary_emb:
            self.rotary_emb = RotaryEmbedding(
                pos_embd_params.rope,
                head_dim=self.qk_rope_head_dim,
                is_neox=pos_embd_params.is_neox,
            )

        if not config.skip_create_weights_in_init:
            self.create_weights()

    def _initialize_split_weights(self):
        """
        Initialize the split weights for split_q_gemm once after the original weights are loaded.
        This avoids doing the weight splitting on every forward pass.
        """
        if not self.split_q_gemm or self._split_weights_initialized:
            return

        # Check if weights are available yet (they might not be during create_weights phase)
        if not hasattr(self.q_b_proj, 'weight') or self.q_b_proj.weight is None:
            return

        # Get the original weight and split it
        original_weight = self.q_b_proj.weight  # [total_dim, q_lora_rank]
        total_nope_dim = self.num_heads * self.qk_nope_head_dim
        total_rope_dim = self.num_heads * self.qk_rope_head_dim

        # Split and store the weight matrices (detach to avoid gradients)
        self._q_nope_weight = original_weight[:total_nope_dim, :].detach()
        self._q_rope_weight = original_weight[total_nope_dim:total_nope_dim +
                                              total_rope_dim, :].detach()
        # Create optimized mixed nope weight that directly produces the concat->view->split result
        # This eliminates the need for concat->view->split operations entirely
        mixed_nope_weight = torch.zeros_like(self._q_nope_weight)

        for head in range(self.num_heads):
            for dim in range(self.qk_nope_head_dim):
                # In the mixed result, position [batch, head, dim] comes from
                # position [batch, head * qk_head_dim + dim] in the concatenated tensor
                mixed_pos = head * self.qk_head_dim + dim

                # This position in the concatenated tensor corresponds to either:
                # - nope data: positions 0 to total_nope_dim-1
                # - rope data: positions total_nope_dim to total_nope_dim+total_rope_dim-1

                if mixed_pos < total_nope_dim:  # It's from nope part
                    # Direct mapping from original nope position
                    original_nope_pos = mixed_pos
                    mixed_nope_weight[head * self.qk_nope_head_dim +
                                      dim, :] = self._q_nope_weight[
                                          original_nope_pos, :]
                else:  # It's from rope part
                    # It's taking rope data, so we need to include rope weight
                    rope_pos = mixed_pos - total_nope_dim
                    mixed_nope_weight[head * self.qk_nope_head_dim +
                                      dim, :] = self._q_rope_weight[rope_pos, :]

        # Store the mixed weight for optimized operations
        self._q_mixed_nope_weight = mixed_nope_weight.detach()

        # Create optimized mixed rope weight that directly produces the rope concat->view->split result
        mixed_rope_weight = torch.zeros_like(self._q_rope_weight)

        for head in range(self.num_heads):
            for dim in range(self.qk_rope_head_dim):
                # In the mixed rope result, position [batch, head, dim] comes from
                # position [batch, head * qk_head_dim + qk_nope_head_dim + dim] in the concatenated tensor
                # (qk_nope_head_dim offset because rope part starts after nope part in each head)
                mixed_pos = head * self.qk_head_dim + self.qk_nope_head_dim + dim

                # This position in the concatenated tensor corresponds to either:
                # - nope data: positions 0 to total_nope_dim-1
                # - rope data: positions total_nope_dim to total_nope_dim+total_rope_dim-1

                if mixed_pos < total_nope_dim:  # It's from nope part
                    # Taking nope data for rope result (this is the mixing!)
                    original_nope_pos = mixed_pos
                    mixed_rope_weight[head * self.qk_rope_head_dim +
                                      dim, :] = self._q_nope_weight[
                                          original_nope_pos, :]
                else:  # It's from rope part
                    # Taking rope data for rope result (normal case)
                    rope_pos = mixed_pos - total_nope_dim
                    mixed_rope_weight[head * self.qk_rope_head_dim +
                                      dim, :] = self._q_rope_weight[rope_pos, :]

        # Store the mixed rope weight for optimized operations
        self._q_mixed_rope_weight = mixed_rope_weight.detach()

        # Create swizzled weight for tk_bmm_fused_matmul only if LLTI is available
        # Use the original _q_nope_weight for backward compatibility
        if LLTI_AVAILABLE:
            self._q_nope_weight_swizzled = swizzle_b_matrix(
                self._q_mixed_nope_weight.t().contiguous(), 4, 8).detach()

            # For transposed GEMM optimization, we need both matrices in K-major format
            # Instead of runtime conversion, let's go back to the standard approach  
            # and avoid the transpose optimization to eliminate extra kernel launches
            # Store weight in the format expected by standard cute GEMM
            weight_temp = self._q_mixed_rope_weight.T.contiguous()  # [1536, 1024] 
            self._q_mixed_rope_weight_cute = weight_temp.t().contiguous().t().detach()  # [1024, 1536] K-major
        else:
            self._q_nope_weight_swizzled = None
            self._q_mixed_rope_weight_cute = None

        # Split bias if present
        if self.q_b_proj.bias is not None:
            self._q_nope_bias = self.q_b_proj.bias[:total_nope_dim].detach()
            self._q_rope_bias = self.q_b_proj.bias[
                total_nope_dim:total_nope_dim + total_rope_dim].detach()
        else:
            self._q_nope_bias = None
            self._q_rope_bias = None

        # Create dummy BMM weights and output buffer for tk_bmm_fused_matmul only if LLTI is available
        if LLTI_AVAILABLE:

            output_per_head = total_nope_dim // self.num_heads
            # Use actual kv_lora_rank instead of hardcoded value
            bmm_output_dim = self.kv_lora_rank

            # Ensure dimensions make sense
            assert self.num_heads > 0, f"num_heads must be positive, got {self.num_heads}"
            assert total_nope_dim % self.num_heads == 0, f"total_nope_dim ({total_nope_dim}) must be divisible by num_heads ({self.num_heads})"

            # Verify we match expected test pattern dimensions

            # Create BMM weights placeholder - will be updated with k_b_proj_trans when available
            # For now, initialize with zeros as placeholder
            self._bmm_weights = torch.zeros(self.num_heads,
                                            output_per_head,
                                            bmm_output_dim,
                                            dtype=self._q_nope_weight.dtype,
                                            device=self._q_nope_weight.device)

            # BMM output buffer for capturing fused BMM result
            # Shape should be [num_heads, batch_size, bmm_output_dim] - batch_size will be updated dynamically
            self._bmm_output_buffer = torch.empty(
                self.num_heads,
                4,
                bmm_output_dim,  # batch_size=4 to match our use case
                dtype=self._q_nope_weight.dtype,
                device=self._q_nope_weight.device)
        else:
            self._bmm_weights = None
            self._bmm_output_buffer = None

        self._split_weights_initialized = True

    def _update_bmm_weights_for_generation(self):
        """
        Update BMM weights to match k_b_proj_trans for forward_generation BMM reuse.
        This allows the BMM output from tk_bmm_fused_matmul to replace the BMM in forward_generation.
        """
        if (hasattr(self, 'k_b_proj_trans') and self.k_b_proj_trans is not None
                and not self._bmm_weights_updated):

            # Reshape k_b_proj_trans to match BMM weights format for fused operation
            # k_b_proj_trans shape: [num_heads, kv_lora_rank, qk_nope_head_dim]
            # Need BMM weights shape: [num_heads, qk_nope_head_dim, kv_lora_rank] (transposed)
            k_b_proj_transposed = self.k_b_proj_trans.transpose(
                1, 2)  # [num_heads, qk_nope_head_dim, kv_lora_rank]

            # Create _bmm_weights with desired stride (65536, 128, 1) using torch.as_strided
            # Current k_b_proj_transposed has stride (65536, 1, 128)
            # We want stride (65536, 128, 1) for better kernel performance
            self._bmm_weights = torch.as_strided(
                k_b_proj_transposed.contiguous(),
                size=k_b_proj_transposed.
                shape,  # Keep same shape [num_heads, qk_nope_head_dim, kv_lora_rank]
                stride=(
                    k_b_proj_transposed.shape[1] *
                    k_b_proj_transposed.shape[2],  # First dim stride
                    k_b_proj_transposed.
                    shape[2],  # Second dim stride = kv_lora_rank
                    1)  # Third dim stride = 1 (contiguous)
            ).clone().detach()

            self._bmm_weights_updated = True

    def create_weights(self):
        # self.mha/mqa has no weights but has states that are related to quant_config,
        # which could be modified after __init__
        self.mha.update_quant_config(self.quant_config)
        self.mqa.update_quant_config(self.quant_config)

        # Note: Split weight initialization is skipped here and will happen during first forward pass
        # when weights are guaranteed to be available

        # k_b_proj_trans's dtype must be consistent with self.kv_b_proj,
        # which can be modified after __init__
        has_fp8_block_scales = (
            self.kv_b_proj.quant_config
            and self.kv_b_proj.quant_config.quant_mode.has_fp8_block_scales())

        mla_weight_dtype = torch.float8_e4m3fn if has_fp8_block_scales else self.dtype
        self.k_b_proj_trans = nn.Parameter(
            torch.empty(
                (self.num_heads, self.kv_lora_rank, self.qk_nope_head_dim),
                dtype=mla_weight_dtype,
            ),
            requires_grad=False,
        )

        self.k_b_proj_trans_dequant = None
        self.v_b_proj_dequant = None
        if has_fp8_block_scales:
            self.k_b_proj_trans_scale = nn.Parameter(
                torch.empty(
                    (
                        self.num_heads,
                        self.kv_lora_rank // 128,
                        self.qk_nope_head_dim // 128,
                    ),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            # This parameter will view into self.kv_b_proj.weight_scale after loading weights.
            # For dummy weight initialization, this parameter is initialized with empty tensor.
            self.v_b_proj_scale = nn.Parameter(
                torch.empty(
                    (
                        self.num_heads,
                        self.v_head_dim // 128,
                        self.kv_lora_rank // 128,
                    ),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            if get_sm_version() == 100:
                assert self.dtype == torch.bfloat16
                self.k_b_proj_trans_dequant = nn.Parameter(
                    torch.empty(
                        (self.num_heads, self.kv_lora_rank,
                         self.qk_nope_head_dim),
                        dtype=self.dtype,
                    ),
                    requires_grad=False,
                )
                self.v_b_proj_dequant = nn.Parameter(
                    torch.empty(
                        (self.num_heads, self.v_head_dim, self.kv_lora_rank),
                        dtype=self.dtype,
                    ),
                    requires_grad=False,
                )
        else:
            self.k_b_proj_trans_scale = None
            self.v_b_proj_scale = None

    def apply_rope(
        self,
        q: torch.Tensor,
        k_pe: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        q = q.view(-1, self.num_heads, self.qk_head_dim)
        q_pe = q[..., self.qk_nope_head_dim:].reshape(
            -1, self.num_heads * self.qk_rope_head_dim)
        q_pe, k_pe = self.rotary_emb(position_ids, [q_pe, k_pe])
        q[..., self.qk_nope_head_dim:] = q_pe.view(-1, self.num_heads,
                                                   self.qk_rope_head_dim)
        return k_pe

    def create_output(self, hidden_states: torch.Tensor):
        num_tokens = hidden_states.shape[0]
        hidden_size = self.o_proj.in_features
        return hidden_states.new_empty([num_tokens, hidden_size],
                                       dtype=hidden_states.dtype)

    def forward_impl(self,
                     position_ids: Optional[torch.Tensor],
                     hidden_states: torch.Tensor,
                     attn_metadata: AttentionMetadata,
                     output: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the MLA module.

        Args:
            position_ids (Optional[torch.IntTensor]): The position IDs.
            hidden_states (torch.Tensor): The hidden states.
            attn_metadata (AttentionMetadata): The attention metadata.
            all_reduce_params (Optional[AllReduceParams]): The all reduce parameters.

        Returns:
            torch.Tensor: The output tensor.
        """
        if self.is_lite:
            compressed_kv, k_pe = self.kv_a_proj_with_mqa(hidden_states).split(
                [self.kv_lora_rank, self.qk_rope_head_dim], -1)
            compressed_kv = self.kv_a_layernorm(compressed_kv)
            q = hidden_states
        else:
            q, compressed_kv, k_pe = self.kv_a_proj_with_mqa(
                hidden_states).split([
                    self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim
                ], -1)

            q, compressed_kv = maybe_execute_in_parallel(
                lambda: self.q_a_layernorm(q),
                lambda: self.kv_a_layernorm(compressed_kv),
                self.ln_events[0],
                self.ln_events[1],
                self.aux_stream,
            )

        # Initialize split weights if needed (regardless of batch size)
        if self.split_q_gemm and not self._split_weights_initialized:
            self._initialize_split_weights()

        if q.shape[0] == 4 and self.split_q_gemm:  #ONLY FOR M dim 4

            def split_gemm_nope():
                self._update_bmm_weights_for_generation()

                current_batch_size = q.shape[0]
                if self._bmm_output_buffer.shape[1] != current_batch_size:
                    self._bmm_output_buffer = torch.empty(
                        self._bmm_weights.shape[0],
                        current_batch_size,
                        self._bmm_weights.shape[2],
                        dtype=q.dtype,
                        device=q.device)

                intermediate_result, _ = torch.ops.llti.tk_bmm_fused_matmul(
                    q, self._q_nope_weight_swizzled, self._bmm_weights,
                    self._bmm_output_buffer)

                # Pure nope GEMM with BMM fusion (concat is now on stream 3)
                return None  # No return value needed from this stream

            def split_gemm_rope():
                # Pure rope GEMM computation (no concat - that's on stream 3)
                if LLTI_AVAILABLE and hasattr(
                        self, '_q_mixed_rope_weight_cute'
                ) and self._q_mixed_rope_weight_cute is not None:
                    # ZERO-COPY transpose optimization using PyTorch .T (view-only operations)
                    # Mathematical identity: A @ B^T = (B @ A^T)^T
                    # weight[1024, 1536] @ q^T[1536, 4] = (q[4, 1536] @ weight^T[1536, 1024])^T
                    
                    # Test if .T creates views (no copy) for both matrices
                    # q.T: [4, 1536] -> [1536, 4] (should be a view)
                    # weight.T: [1024, 1536] -> [1536, 1024] (should be a view)  
                    result_transposed = torch.ops.llti.cute_gemm_bf16(
                        self._q_mixed_rope_weight_cute.T,  # [1536, 1024] - view operation
                        q.T,                               # [1536, 4] - view operation
                        None,  # C: let cute GEMM create output
                        None,  # bias: handle separately due to different convention
                        True,  # fdl: fast descriptor loading
                        4,  # fdl_count: default
                        8  # dma_stage: default
                    )
                    
                    # Final transpose back: [1024, 4] -> [4, 1024] (view operation)
                    result = result_transposed.T
                    
                    # Add bias manually since cute GEMM has different bias convention
                    if self._q_rope_bias is not None:
                        result = result + self._q_rope_bias
                    # Ensure contiguous layout for downstream TensorRT-LLM operations
                    # TensorRT-LLM attention expects stride()[last_dim] == 1
                    rope_result = result.contiguous()
                else:
                    # Fallback to standard linear implementation
                    rope_result = torch.nn.functional.linear(
                        q, self._q_mixed_rope_weight, self._q_rope_bias)

                return rope_result

            def split_concat():
                # Pure concat operation for stream 3
                return torch.concat([compressed_kv, k_pe], dim=-1)
            
            try:

                # # Run all 3 operations in parallel on separate streams
                _, q_rope_mixed, latent_cache = maybe_execute_in_parallel_3streams(
                    split_gemm_nope,  # Stream 1 (default): Wq nope GEMM (BMM fusion only)
                    split_gemm_rope,  # Stream 2 (aux_stream): Wq rope GEMM (pure GEMM)
                    split_concat,     # Stream 3 (concat_stream): concat operation
                    self.ln_events[0],  # Event for stream 1
                    self.ln_events[1],  # Event for stream 2  
                    self.ln_events[2],  # Event for stream 3
                    self.aux_stream,    # aux_stream1 for rope GEMM
                    self.concat_stream, # aux_stream2 for concat
                )
                # _ = split_gemm_nope()
                # q_rope_mixed = split_gemm_rope()
                # latent_cache = split_concat()

            except Exception:
                raise
        else:
            # Fallback to 2-stream execution when split_q_gemm is disabled
            q, latent_cache = maybe_execute_in_parallel(
                lambda: self.q_b_proj(q),  #Wq gemm + Wqr gemm
                lambda: torch.concat([compressed_kv, k_pe], dim=-1),
                self.ln_events[0],
                self.ln_events[1],
                self.aux_stream,
            )

        # split q, k, v into context and gen batches
        num_contexts = attn_metadata.num_contexts
        num_generations = attn_metadata.num_generations
        num_ctx_tokens = attn_metadata.num_ctx_tokens
        num_tokens = attn_metadata.num_tokens

        if q.shape[
                0] == 4 and self.split_q_gemm and self.use_fused_bmm_for_generation:
            pass
        else:
            assert q.shape[
                0] == num_tokens, f"Expect q.shape[0] to be {num_tokens}, but got {q.shape[0]}"

        if num_contexts > 0:
            q_ctx = q[:num_ctx_tokens, ...]
            compressed_kv_ctx = compressed_kv[:num_ctx_tokens, ...]
            k_pe_ctx = k_pe[:num_ctx_tokens, ...]
            latent_cache_ctx = latent_cache[:num_ctx_tokens, ...]
            if self.apply_rotary_emb:
                assert position_ids is not None
                k_pe_ctx = self.apply_rope(q_ctx, k_pe_ctx, position_ids)

            attn_output_context = self.forward_context(
                q_ctx,
                compressed_kv_ctx,
                k_pe_ctx,
                attn_metadata,
                latent_cache_ctx,
                output=output if num_generations == 0 else None)
            if num_generations == 0:
                return attn_output_context
        else:
            attn_output_context = None

        if num_generations > 0:
            if q.shape[
                    0] == 4 and self.split_q_gemm and self.use_fused_bmm_for_generation:
                q_gen = q_rope_mixed
            else:
                q_gen = q[num_ctx_tokens:, ...]
            compressed_kv_gen = compressed_kv[num_ctx_tokens:, ...]
            k_pe_gen = k_pe[num_ctx_tokens:, ...]
            latent_cache_gen = latent_cache[num_ctx_tokens:, ...]
            if self.apply_rotary_emb:
                assert position_ids is not None
                k_pe_gen = self.apply_rope(q_gen, k_pe_gen, position_ids)

            attn_output_gen = self.forward_generation(
                q_gen,
                compressed_kv_gen,
                k_pe_gen,
                attn_metadata,
                latent_cache_gen,
                output=output if num_contexts == 0 else None)
            if num_contexts == 0:
                return attn_output_gen
        else:
            attn_output_gen = None

        # release pytorch activation memory
        q = None
        compressed_kv = None
        k_pe = None

        assert attn_output_context is not None and attn_output_gen is not None
        assert (
            len(attn_output_context.shape) == 2
        ), f"attn_output_context must be rank 2, not {len(attn_output_context.shape)}"
        assert (
            len(attn_output_gen.shape) == 2
        ), f"attn_output_gen must be rank 2, not {len(attn_output_gen.shape)}"
        output = output if output is not None else torch.empty(
            (num_tokens, attn_output_context.shape[1]),
            dtype=attn_output_context.dtype,
            device=attn_output_context.device)
        output[:attn_output_context.shape[0], :] = attn_output_context
        output[attn_output_context.shape[0]:, :] = attn_output_gen
        attn_output_context = None
        attn_output_gen = None
        return output

    def _maybe_concat_qkv(self, q, k, v):
        if k is not None and v is not None and self.support_fused_qkv:
            qkv = torch.concat([q, k, v], dim=-1)
            q, k, v = qkv, None, None
        return q, k, v

    def forward_context_default(
            self,
            q: torch.Tensor,
            compressed_kv: torch.Tensor,
            k_pe: torch.Tensor,
            attn_metadata: AttentionMetadata,
            latent_cache: Optional[torch.Tensor] = None,
            output: Optional[torch.Tensor] = None) -> torch.Tensor:
        kv = self.kv_b_proj(compressed_kv)
        k_nope, v = kv.split(
            [
                self.num_heads * self.qk_nope_head_dim,
                self.num_heads * self.v_head_dim
            ],
            -1,
        )

        k = torch.empty_like(q).view(-1, self.num_heads, self.qk_head_dim)
        k[..., :self.qk_nope_head_dim] = k_nope.view(-1, self.num_heads,
                                                     self.qk_nope_head_dim)
        if self.apply_rotary_emb:
            k[..., self.qk_nope_head_dim:] = k_pe.view(-1, 1,
                                                       self.qk_rope_head_dim)
        k = k.view(-1, self.num_heads * self.qk_head_dim)

        # May concat q(including q_pe), k + k_pe, v together
        q, k, v = self._maybe_concat_qkv(q, k, v)

        # out_scale = getattr(self.o_proj, "inv_input_scale", None)
        out_scale = None  # Currently we use BF16 MHA for context phase

        attn_output = self.mha.forward(
            q,
            k,
            v,
            attn_metadata,
            attention_input_type=AttentionInputType.context_only,
            latent_cache=latent_cache,
            out_scale=out_scale,
            output=output,
        )

        return attn_output

    def forward_context_with_cached_kv(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert latent_cache is not None
        trtllm_attention = cast(TrtllmAttention, self.mha)

        # apply RoPE, append compressed_kv + k_pe to paged kv cache and assign q_pe to q
        trtllm_attention.mla_rope_append_paged_kv_assign_q(
            q, latent_cache, attn_metadata)

        # copy full_compressed_kv and full_k_pe from paged kv cache
        full_compressed_kv, full_k_pe = trtllm_attention.load_paged_kv_cache_for_mla(
            attn_metadata, q.dtype)
        assert full_compressed_kv.shape[
            0] == attn_metadata.num_ctx_cached_tokens + attn_metadata.num_ctx_tokens
        assert full_compressed_kv.shape[1] == self.kv_lora_rank
        assert full_k_pe.shape[
            0] == attn_metadata.num_ctx_cached_tokens + attn_metadata.num_ctx_tokens
        assert full_k_pe.shape[1] == self.qk_rope_head_dim
        assert full_compressed_kv.is_contiguous()
        assert full_k_pe.is_contiguous()

        # compute full_k_nope and full_v from full_compressed_kv
        full_kv = self.kv_b_proj(full_compressed_kv)
        full_k_nope, full_v = full_kv.split(
            [
                self.num_heads * self.qk_nope_head_dim,
                self.num_heads * self.v_head_dim
            ],
            -1,
        )
        full_k_nope = full_k_nope.view(-1, self.num_heads,
                                       self.qk_nope_head_dim)
        full_v = full_v.view(-1, self.num_heads, self.v_head_dim)

        # build paged_full_kv
        tokens_per_block = attn_metadata.kv_cache_manager.tokens_per_block
        # paged_full_kv will be initialized to 0 in the kernel to avoid NaN
        paged_full_kv = torch.empty([
            attn_metadata.num_contexts, 2,
            (attn_metadata.max_ctx_kv_len + tokens_per_block - 1) //
            tokens_per_block, self.num_heads, tokens_per_block,
            max(self.qk_nope_head_dim + self.qk_rope_head_dim, self.v_head_dim)
        ],
                                    dtype=q.dtype,
                                    device=q.device)
        mla_context_kv_cache_block_offsets = trtllm_attention.set_paged_kv_cache_for_mla(
            paged_full_kv,
            full_k_nope,
            full_v,
            full_k_pe,
            attn_metadata,
        )

        # release pytorch activation memory
        full_compressed_kv = None
        full_k_pe = None
        full_kv = None
        full_k_nope = None
        full_v = None

        # out_scale = getattr(self.o_proj, "inv_input_scale", None)
        out_scale = None  # Currently we use BF16 MHA for context phase

        attn_output = self.mha.forward(
            q,
            None,
            None,
            attn_metadata,
            attention_input_type=AttentionInputType.context_only,
            latent_cache=None,
            out_scale=out_scale,
            mla_context_paged_kv=paged_full_kv,
            mla_context_kv_cache_block_offsets=
            mla_context_kv_cache_block_offsets,
            output=output,
        )

        return attn_output

    def forward_context_with_chunked_prefill(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        latent_cache: torch.
        Tensor,  # compressed_kv + k_pe [context_tokens, 1, lora_size + rope_size]
        attn_metadata: TrtllmAttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        trtllm_attention = cast(TrtllmAttention, self.mha)
        # apply RoPE, append compressed_kv + k_pe to paged kv cache and assign q_pe to q
        trtllm_attention.mla_rope_append_paged_kv_assign_q(
            q, latent_cache, attn_metadata)

        # determine the number of loop
        # currently we assume that the chunk size is the same as the max_num_tokens
        chunk_size = attn_metadata.runtime_features.chunk_size
        chunked_loop_num = attn_metadata.chunked_loop_num

        # [toal_token_q, num_heads, 2] -> [toal_token_q, num_heads] float2
        self.softmax_stats_tensor = torch.empty(
            (attn_metadata.num_ctx_tokens, self.num_heads, 2),
            dtype=torch.float,
            device='cuda',
        )
        self.temp_softmax_stats_tensor = torch.empty(
            (attn_metadata.num_ctx_tokens, self.num_heads, 2),
            dtype=torch.float,
            device='cuda',
        )
        if output is None:
            attn_output = q.new_empty(
                (q.size(0), self.num_heads * self.v_head_dim), dtype=q.dtype)
        else:
            attn_output = output
        temp_attn_output = q.new_empty(
            (q.size(0), self.num_heads * self.v_head_dim), dtype=q.dtype)

        # use fake cached_cu_seq_len for chunked loop
        origin_kv_lens_cuda_runtime = attn_metadata.kv_lens_cuda_runtime
        origin_kv_lens_runtime = attn_metadata.kv_lens_runtime

        for loop_idx in range(chunked_loop_num):
            # {b, chunked_unit_size, h, kv_lora_rank + qk_rope_head_dim} zero padded
            # fetch `loop_idx` chunk from kv cache
            temp_cu_chunked_seq_len = attn_metadata.cu_chunked_seq_len[loop_idx]
            total_ctx_chunked_tokens = attn_metadata.host_cu_chunked_seq_len[
                loop_idx, attn_metadata.num_contexts]
            chunked_compressed_kv, chunked_k_pe = trtllm_attention.load_chunked_kv_cache_for_mla(
                metadata=attn_metadata,
                chunked_idx=loop_idx,
                num_ctx_cached_tokens=total_ctx_chunked_tokens,
                cu_chunked_seq_len=temp_cu_chunked_seq_len,
                out_dtype=q.dtype)

            # up proj to uncompressed kv
            # [tokens, 2, h, kv_dim], without rope_dim
            chunked_kv = self.kv_b_proj(chunked_compressed_kv)

            # build full_kv
            # full_kv {B, 2, chunk_size / tokens_per_block, h, tokens_per_block, kv_dim + rope_dim}
            tokens_per_block = attn_metadata.kv_cache_manager.tokens_per_block
            full_kv = torch.zeros([
                attn_metadata.num_contexts, 2,
                (chunk_size + tokens_per_block - 1) // tokens_per_block,
                self.num_heads, tokens_per_block,
                max(self.qk_nope_head_dim + self.qk_rope_head_dim,
                    self.v_head_dim)
            ],
                                  dtype=q.dtype,
                                  device=q.device)
            mla_kv_cache_block_offsets = trtllm_attention.set_chunked_kv_cache_for_mla(
                full_kv,
                chunked_kv,
                chunked_k_pe,
                cu_chunked_seq_len=temp_cu_chunked_seq_len,
                cached=True,
                metadata=attn_metadata)

            # copy chunked_seq_len to replace kv_lens_runtime
            attn_metadata.kv_lens_runtime = attn_metadata.host_chunked_seq_len[
                loop_idx]
            attn_metadata.kv_lens_cuda_runtime = attn_metadata.chunked_seq_len[
                loop_idx]
            out_scale = None
            # do not apply mask for attention within loop
            temp_attn_output = self.mha.forward(
                q,
                None,
                None,
                attn_metadata,
                attention_input_type=AttentionInputType.context_only,
                latent_cache=None,
                out_scale=out_scale,
                attention_mask=PredefinedAttentionMask.FULL,
                mla_context_paged_kv=full_kv,
                mla_context_kv_cache_block_offsets=mla_kv_cache_block_offsets,
                softmax_stats_tensor=self.temp_softmax_stats_tensor,
                output=temp_attn_output,
            )
            # merge attn result
            temp_merge_op = attn_metadata.merge_op_tensor[loop_idx]
            trtllm_attention.merge_attention_for_mla(
                attn_output, temp_attn_output, self.softmax_stats_tensor,
                self.temp_softmax_stats_tensor, temp_merge_op, attn_metadata)

        # deal with the uncached kv
        kv = self.kv_b_proj(compressed_kv)
        _, k_pe = latent_cache.view([
            -1, self.kv_lora_rank + self.qk_rope_head_dim
        ]).split([self.kv_lora_rank, self.qk_rope_head_dim], -1)
        k_pe = k_pe.contiguous()
        # final round of attention

        # out_scale = getattr(self.o_proj, "inv_input_scale", None)
        out_scale = None  # Currently we use BF16 MHA for context phase

        tokens_per_block = attn_metadata.kv_cache_manager.tokens_per_block
        full_kv = torch.zeros([
            attn_metadata.num_contexts, 2,
            (attn_metadata.max_ctx_seq_len + tokens_per_block - 1) //
            tokens_per_block, self.num_heads, tokens_per_block,
            max(self.qk_nope_head_dim + self.qk_rope_head_dim, self.v_head_dim)
        ],
                              dtype=q.dtype,
                              device=q.device)
        mla_kv_cache_block_offsets = trtllm_attention.set_chunked_kv_cache_for_mla(
            full_kv,
            kv,
            k_pe,
            cu_chunked_seq_len=None,
            cached=False,
            metadata=attn_metadata)
        # copy q_lens to replace kv_lens_runtime
        attn_metadata.kv_lens_runtime = attn_metadata.prompt_lens_cpu_runtime
        attn_metadata.kv_lens_cuda_runtime = attn_metadata.prompt_lens_cuda_runtime
        temp_attn_output = self.mha.forward(
            q,
            None,
            None,
            attn_metadata,
            attention_input_type=AttentionInputType.context_only,
            latent_cache=None,
            out_scale=out_scale,
            mla_context_paged_kv=full_kv,
            mla_context_kv_cache_block_offsets=mla_kv_cache_block_offsets,
            softmax_stats_tensor=self.temp_softmax_stats_tensor,
            output=temp_attn_output,
        )
        temp_merge_op = attn_metadata.merge_op_tensor[chunked_loop_num]
        trtllm_attention.merge_attention_for_mla(attn_output, temp_attn_output,
                                                 self.softmax_stats_tensor,
                                                 self.temp_softmax_stats_tensor,
                                                 temp_merge_op, attn_metadata)
        # copy back kv_lens_runtime and kv_lens_cuda_runtime
        attn_metadata.kv_lens_runtime = origin_kv_lens_runtime
        attn_metadata.kv_lens_cuda_runtime = origin_kv_lens_cuda_runtime

        return attn_output

    def forward_context(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
        latent_cache: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(self.mha, TrtllmAttention):
            assert isinstance(attn_metadata, TrtllmAttentionMetadata)
            trtllm_attention = cast(TrtllmAttention, self.mha)
            if trtllm_attention.is_chunked_prefill_for_mla_context(
                    attn_metadata):
                return self.forward_context_with_chunked_prefill(
                    q, compressed_kv, latent_cache, attn_metadata, output)
            elif trtllm_attention.has_cached_kv_for_mla_context(attn_metadata):
                return self.forward_context_with_cached_kv(
                    q, latent_cache, attn_metadata, output)
        return self.forward_context_default(q, compressed_kv, k_pe,
                                            attn_metadata, latent_cache, output)

    def forward_generation(
            self,
            q: torch.Tensor,
            compressed_kv: torch.Tensor,
            k_pe: torch.Tensor,
            attn_metadata: AttentionMetadata,
            latent_cache: Optional[torch.Tensor] = None,
            output: Optional[torch.Tensor] = None) -> torch.Tensor:

        num_tokens = q.shape[0]

        if not (q.shape[0] == 4 and self.split_q_gemm
                and self.use_fused_bmm_for_generation):
            q_nope, q_pe = q.view([-1, self.num_heads, self.qk_head_dim]).split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        else:
            q_pe = q.view(q.shape[0], self.num_heads, self.qk_rope_head_dim)

        # fused_q contains 1) the result of the following bmm with shape [num_tokens, num_heads, kv_lora_rank]
        # 2) rope(q_pe) with shape [num_tokens, num_heads, qk_rope_head_dim]. rope is applied inside AttentionOp
        fused_q = torch.empty(
            [
                num_tokens, self.num_heads,
                (self.kv_lora_rank + self.qk_rope_head_dim)
            ],
            dtype=q.dtype,
            device=q.device,
        )

        if self.k_b_proj_trans.dtype == torch.bfloat16:

            # FUSED BMM GENERATION: Use cached BMM output if available from tk_bmm_fused_matmul
            # use_cached_bmm = (
            #     self.use_fused_bmm_for_generation
            #     and self._cached_bmm_output is not None and num_tokens ==
            #     4  # Only when batch size is 4 (matching forward_impl condition)
            # )

            use_cached_bmm = (
                self.use_fused_bmm_for_generation and num_tokens ==
                4  # Only when batch size is 4 (matching forward_impl condition)
            )

            if use_cached_bmm:

                fused_q[..., :self.
                        kv_lora_rank] = self._bmm_output_buffer.transpose(0, 1)

            else:
                # Standard BMM path: Use torch.nn.functional.linear + torch.ops.trtllm.bmm_out
                # [num_heads, num_tokens, self.kv_lora_rank]
                q_nope_out = fused_q[..., :self.kv_lora_rank].transpose(0, 1)
                # [num_heads, num_tokens, self.qk_nope_head_dim]
                q_nope_t = q_nope.transpose(0, 1)

                torch.ops.trtllm.bmm_out(
                    q_nope_t,  # Wk^T
                    self.k_b_proj_trans.transpose(1, 2),
                    q_nope_out)

        elif self.k_b_proj_trans.dtype == torch.float8_e4m3fn:
            # [num_heads, num_tokens, self.kv_lora_rank]
            q_nope_out = fused_q[..., :self.kv_lora_rank].transpose(0, 1)

            fp8_block_scaling_bmm_out(
                q_nope,
                self.k_b_proj_trans,
                self.k_b_proj_trans_scale,
                q_nope_out,
                self.k_b_proj_trans_dequant,
            )
        else:
            raise NotImplementedError(
                f"Missing bmm impl for dtype: {self.k_b_proj_trans.dtype}.")

        if self.apply_rotary_emb:
            fused_q[..., self.kv_lora_rank:] = q_pe
        fused_q = fused_q.view([
            num_tokens,
            self.num_heads * (self.kv_lora_rank + self.qk_rope_head_dim)
        ])

        # out_scale = getattr(self.o_proj, "inv_input_scale", None)
        out_scale = None  # Although we use FP8 MLA for generation phase, the output is still in BF16

        attn_out_latent = self.mqa.forward(
            fused_q,
            None,
            None,
            attn_metadata,
            attention_input_type=AttentionInputType.generation_only,
            out_scale=out_scale,
            latent_cache=latent_cache,  # kvcache and k_pe
            q_pe=q_pe,  # used by `invokeMLARopeGeneration`
        )
        fused_q = None

        assert (attn_out_latent.shape[0] == q.shape[0] and
                attn_out_latent.shape[1] == self.num_heads * self.kv_lora_rank)

        # [seq, num_heads, kv_lora_rank]
        attn_out_latent = attn_out_latent.view(
            [-1, self.num_heads, self.kv_lora_rank])

        # [seq, num_heads * v_head_dim]
        output = output if output is not None else torch.empty(
            [num_tokens, self.num_heads * self.v_head_dim],
            dtype=attn_out_latent.dtype,
            device=attn_out_latent.device)

        attn_output = output.view([num_tokens, self.num_heads, self.v_head_dim])

        if self.v_b_proj.dtype == torch.bfloat16:
            # [num_heads, seq, kv_lora_rank] x [num_heads, kv_lora_rank, v_head_dim]
            # -> [num_heads, seq, v_head_dim]
            torch.ops.trtllm.bmm_out(attn_out_latent.transpose(0, 1),
                                     self.v_b_proj.transpose(1, 2),
                                     attn_output.transpose(0, 1))
        elif self.v_b_proj.dtype == torch.float8_e4m3fn:
            fp8_block_scaling_bmm_out(
                attn_out_latent,
                self.v_b_proj,
                self.v_b_proj_scale,
                attn_output.transpose(0, 1),
                self.v_b_proj_dequant,
            )
        else:
            raise NotImplementedError(
                f"Missing bmm impl for dtype: {self.v_b_proj.dtype}.")

        return output

    def forward(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams] = None,
    ) -> torch.Tensor:

        attn_output = self.create_output(hidden_states)
        if self.register_to_config:
            torch.ops.trtllm.mla_custom_op_inplace(hidden_states, position_ids,
                                                   self.layer_idx_str,
                                                   attn_output)
        else:
            self.forward_impl(position_ids,
                              hidden_states,
                              attn_metadata,
                              output=attn_output)
        attn_output = self.o_proj(attn_output,
                                  all_reduce_params=all_reduce_params)
        return attn_output
