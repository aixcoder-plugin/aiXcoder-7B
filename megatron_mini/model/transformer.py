# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Transformer."""
import math
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union
from megatron_mini import get_args, core
from .module import MegatronModule
from megatron_mini.core import mpu, tensor_parallel
from megatron_mini.model.module import AttnMaskType

def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True, file=sys.stderr)
    else:
        print(message, flush=True, file=sys.stderr)

try:
    from einops import rearrange
except ImportError:
    rearrange = None

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    print_rank_0(f"WARNING: FlashAttention is not available")
    flash_attn_varlen_func = None

try:
    from flash_attn.flash_attn_interface import flash_attn_with_kvcache
except ImportError:
    print_rank_0(f"WARNING: FlashAttention is not available")
    flash_attn_with_kvcache = None


class FusedScaleMaskSoftmax(nn.Module):
    def __init__(
        self,
        input_in_fp16,
        input_in_bf16,
        attn_mask_type,
        scaled_masked_softmax_fusion,
        mask_func,
        softmax_in_fp32,
        scale,
    ):
        super(FusedScaleMaskSoftmax, self).__init__()
        self.input_in_fp16 = input_in_fp16
        self.input_in_bf16 = input_in_bf16
        assert not (
            self.input_in_fp16 and self.input_in_bf16
        ), "both fp16 and bf16 flags cannot be active at the same time."
        self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16
        self.attn_mask_type = attn_mask_type
        self.scaled_masked_softmax_fusion = scaled_masked_softmax_fusion
        self.mask_func = mask_func
        self.softmax_in_fp32 = softmax_in_fp32
        self.scale = scale

        assert (
            self.scale is None or softmax_in_fp32
        ), "softmax should be in fp32 when scaled"

    def forward(self, input, mask):
        # [b, np, sq, sk]
        assert input.dim() == 4

        if self.input_in_float16 and self.softmax_in_fp32:
            input = input.float()

        if self.scale is not None:
            input = input * self.scale
        mask_output = self.mask_func(input, mask) if mask is not None else input
        probs = torch.nn.Softmax(dim=-1)(mask_output)

        if self.input_in_float16 and self.softmax_in_fp32:
            if self.input_in_fp16:
                probs = probs.half()
            else:
                probs = probs.bfloat16()

        return probs


def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


def _args_to_kwargs():
    args = get_args()

    common_kwargs = {
        "params_dtype": args.params_dtype,
        "use_cpu_initialization": args.use_cpu_initialization,
        "perform_initialization": False,
        "gradient_accumulation_fusion": False,
        "sequence_parallel_enabled": args.sequence_parallel,
    }
    return common_kwargs


class FlashSelfAttention(torch.nn.Module):
    def __init__(self, causal=False, softmax_scale=None):
        super().__init__()
        assert flash_attn_varlen_func is not None, ('Please install FlashAttention first, '
                                                      'e.g., with pip install flash-attn')
        assert rearrange is not None, 'Please install einops first, e.g., with pip install einops'
        self.causal = causal
        self.softmax_scale = softmax_scale

    def forward(self, q, k, v):
        """Implements the softmax like attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """
        assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q,k,v)))
        assert all((i.is_cuda for i in (q,k,v)))

        if flash_attn_with_kvcache is not None:

            """
            q: (batch_size, seqlen_q, nheads, headdim)
            k_cache: (batch_size, seqlen_kv, nheads_k, headdim)
            v_cache: (batch_size, seqlen_kv, nheads_k, headdim)

            we do not pass k and v to flash_attn_with_kvcache, because our k and v are packed with kv cache
            
            """
            context = flash_attn_with_kvcache(
                q,
                k_cache=k,
                v_cache=v,
                k=None,
                v=None,
                cache_seqlens=None,
                softmax_scale=self.softmax_scale,
                causal=True,
            )
            return context


        else:
            batch_size, seqlen_q = q.shape[0], q.shape[1]
            seqlen_k =k.shape[1]
            q, k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [q, k, v]]
            is_causal = seqlen_q == seqlen_k
            
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q.device)
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=k.device)
            output = flash_attn_varlen_func(
                q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k,
                0.0,
                softmax_scale=self.softmax_scale, causal=is_causal
            )
            output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
            return output


class CoreAttention(MegatronModule):

    def __init__(self, layer_number,
                 attn_mask_type=AttnMaskType.padding):
        super(CoreAttention, self).__init__()
        args = get_args()
        self.fp16 = args.fp16
        self.bf16 = args.bf16

        self.apply_query_key_layer_scaling = args.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = False
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.sequence_parallel = args.sequence_parallel

        projection_size = args.kv_channels * args.num_attention_heads

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = core.utils.divide(projection_size,
                                                           world_size)
        self.hidden_size_per_attention_head = core.utils.divide(
            projection_size, args.num_attention_heads)
        self.num_attention_heads_per_partition = core.utils.divide(
            args.num_attention_heads, world_size)

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16, self.bf16,
            self.attn_mask_type,
            args.masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff)

    def forward(self, query_layer, key_layer,
                value_layer, attention_mask):

        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2],
                                       output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3],
                                   output_size[0] * output_size[1], -1)

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = mpu.get_global_memory_buffer().get_tensor(
            (output_size[0]*output_size[1], output_size[2], output_size[3]),
            query_layer.dtype, "mpu")

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer.transpose(0, 1),   # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0, alpha=(1.0/self.norm_factor))

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ===========================
        # Attention probs
        # ===========================

        if os.getenv('AIXCODER_DEBUG') == 'ON':
            mp_0 = query_layer.shape[1]
            stats = torch.stack([
                torch.mean(attention_scores[:,:mp_0]).to(torch.float32), 
                torch.std(attention_scores[:,:mp_0]).to(torch.float32), 
                torch.max(attention_scores[:,:mp_0]).to(torch.float32)
            ]).detach().cpu().numpy()

            print_rank_0(
                f"\nAttention - scores before softmax".ljust(40) + f": {stats}, {attention_scores[:,:mp_0].dtype}, {attention_scores[:,:mp_0].shape}"
            )

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)

        if os.getenv('AIXCODER_DEBUG') == 'ON':
            mp_0 = query_layer.shape[1]
            stats = torch.stack([
                torch.mean(attention_probs[:,:mp_0]).to(torch.float32), 
                torch.std(attention_probs[:,:mp_0]).to(torch.float32), 
                torch.max(attention_probs[:,:mp_0]).to(torch.float32)
            ]).detach().cpu().numpy()

            print_rank_0(
                f"\nAttention_probs".ljust(40) + f": {stats}, {attention_probs[:,:mp_0].dtype}, {attention_probs[:,:mp_0].shape}"
            )
        

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1),
                       value_layer.size(2),
                       query_layer.size(0),
                       value_layer.size(3))

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0),
                                       output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                               output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


"""

Implementation for LLaMA

"""


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# @torch.jit.script
def apply_rotary_pos_emb(x, cos, sin):

    # Handle a possible sequence length mismatch in between q and k
    cos = cos[:x.shape[0], :, :, :]
    sin = sin[:x.shape[0], :, :, :]

    part_1 = x * cos

    x1, x2 = x.chunk(2, dim=-1)
    part_2 = torch.cat((-x2, x1), dim=-1) * sin
    return part_1 + part_2


class RotaryEmbedding(MegatronModule):
    def __init__(self, seq_dimension=0, rope_theta=10000, *_, **__):
        super().__init__()
        args = get_args()
        self.args = args
        self.seq_dimension = seq_dimension
        dim_model = args.hidden_size // args.num_attention_heads

        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim_model, 2).float() / dim_model))

        # persistent: whether the buffer is part of this module's :attr:`state_dict`
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._seq_len_cached = self.args.seq_length
        t = torch.arange(
            self._seq_len_cached, dtype=torch.float32
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos_cached = emb.cos()[:, None, None, :]
        self._sin_cached = emb.sin()[:, None, None, :]

    def _update_cos_sin_tables(self, x):
        seq_len = x.shape[self.seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if (
            seq_len != self._seq_len_cached
            or self._cos_cached.device != x.device
            or self._cos_cached.dtype != x.dtype
        ):
            self._seq_len_cached = seq_len
            t = torch.arange(
                x.shape[self.seq_dimension], device=x.device, dtype=torch.float32
            )
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(x.dtype))
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            if self.seq_dimension == 1:
                self._cos_cached = emb.cos()[None, :, None, :].to(x.dtype)
                self._sin_cached = emb.sin()[None, :, None, :].to(x.dtype)
            elif self.seq_dimension == 0:
                self._cos_cached = emb.cos()[:, None, None, :].to(x.dtype)
                self._sin_cached = emb.sin()[:, None, None, :].to(x.dtype)
            else:
                raise NotImplementedError

        return self._cos_cached, self._sin_cached

    def forward(
        self, query_layer: torch.Tensor, key_layer: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            query_layer: [seq_len, bsz, local_num_heads, heads_dim]
        """
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(
            key_layer
        )

        return (
            apply_rotary_pos_emb(query_layer, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(key_layer, self._cos_cached, self._sin_cached),
        )

    def set_devices_dtype(self, x):
        self._cos_cached = self._cos_cached.to(x.dtype).to(x.device)
        self._sin_cached = self._sin_cached.to(x.dtype).to(x.device)
    
    def get_freqs_cis(self, h_shape, start_pos=0):
        seq_len = h_shape[self.seq_dimension]
        if self.seq_dimension == 1:
            return torch.stack((self._cos_cached[:, start_pos: start_pos + seq_len], self._sin_cached[:, start_pos: start_pos + seq_len]), dim=0)
        elif self.seq_dimension == 0:
            return torch.stack((self._cos_cached[start_pos: start_pos + seq_len], self._sin_cached[start_pos: start_pos + seq_len]), dim=0)
        else:
            raise NotImplementedError

    @staticmethod
    # @torch.jit.script
    def apply_rotary(
        query_layer: torch.Tensor, key_layer: torch.Tensor, freqs_cis: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        cos = freqs_cis[0]
        sin = freqs_cis[1]

        # handle short query 
        q_part_1 = query_layer * cos[-query_layer.shape[0]:]
        q_x1, q_x2 = query_layer.chunk(2, dim=-1)
        q_part_2 = torch.cat((-q_x2, q_x1), dim=-1) * sin[-query_layer.shape[0]:]

        k_part_1 = key_layer * cos
        k_x1, k_x2 = key_layer.chunk(2, dim=-1)
        k_part_2 = torch.cat((-k_x2, k_x1), dim=-1) * sin

        return q_part_1 + q_part_2, k_part_1 + k_part_2


class LinearScalingRotaryEmbedding(RotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, seq_dimension=0, rope_theta=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(seq_dimension, rope_theta, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos_cached = emb.cos()[:, None, None, :]
        self._sin_cached = emb.sin()[:, None, None, :]


class LLaMAttention(MegatronModule):
    def __init__(self, init_method,
                 output_layer_init_method, layer_number):
        super().__init__()
        args = get_args()

        self.params_dtype = args.params_dtype
        mp_world_size = mpu.get_tensor_model_parallel_world_size()
        

        self.n_local_heads = core.utils.divide(
            args.num_attention_heads, mp_world_size)
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.local_num_kv_heads = core.utils.divide(
            args.num_kv_heads, mp_world_size)

        self.attention_head_type = args.attention_head_type
        self.sequence_parallel = args.sequence_parallel

        if self.attention_head_type == "multihead":
            self.query_key_value = tensor_parallel.ColumnParallelLinear(
                args.hidden_size,
                3 * args.hidden_size,
                gather_output=False,
                init_method=init_method,
                bias=False,
                skip_bias_add=True,
                async_tensor_model_parallel_allreduce=args.async_tensor_model_parallel_allreduce,
                **_args_to_kwargs())
            
            self.core_attention = CoreAttention(layer_number,
                                                AttnMaskType.causal)
            
            self.cache_k = torch.zeros(
                (args.max_position_embeddings, args.micro_batch_size, self.n_local_heads, self.head_dim)
            ).cuda()
            self.cache_v = torch.zeros(
                (args.max_position_embeddings, args.micro_batch_size, self.n_local_heads, self.head_dim)
            ).cuda()
        elif self.attention_head_type == "groupedquery":
            self.query_key_value = tensor_parallel.ColumnParallelLinear(
                args.hidden_size,
                args.hidden_size + args.num_kv_heads * 2 * self.head_dim,
                gather_output=False,
                init_method=init_method,
                bias=False,
                skip_bias_add=True,
                async_tensor_model_parallel_allreduce=args.async_tensor_model_parallel_allreduce,
                **_args_to_kwargs())
            cached_kv_heads_num = core.utils.divide(
                args.num_kv_heads, mp_world_size)
            
            self.core_attention = CoreAttention(layer_number,
                                                AttnMaskType.causal)
            
            # shape: [seq_len, bsz, local_kv_head_num=group_size, key_or_value=1, head_dim]
            self.cache_k = torch.zeros(
                (args.max_position_embeddings, args.micro_batch_size, cached_kv_heads_num, 1, self.head_dim)
            ).cuda()
            self.cache_v = torch.zeros(
                (args.max_position_embeddings, args.micro_batch_size, cached_kv_heads_num, 1, self.head_dim)
            ).cuda()
        else:
            raise ValueError(f"attention type was Wrong with {self.attention_head_type} in llama")

        self.wo = tensor_parallel.RowParallelLinear(
            args.hidden_size,
            args.hidden_size,
            params_dtype=self.params_dtype,
            input_is_parallel=True if mp_world_size > 1 else False,
            init_method=output_layer_init_method,
            perform_initialization=False,
            use_cpu_initialization=True,
            bias=False,
            skip_bias_add=True)
        
        self.use_flash_attn = args.use_flash_attn
        
        self.core_attention_flash = None
        if self.use_flash_attn and flash_attn_varlen_func is not None and rearrange is not None:
            self.core_attention_flash = FlashSelfAttention(
                causal=True
            )
        else:
            self.use_flash_attn = False

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, attention_mask: Optional[torch.Tensor]):
        seqlen, bsz, _ = x.shape

        if self.attention_head_type == 'multihead':
            # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
            mixed_x_layer, _ = self.query_key_value(x)

            # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                (self.n_local_heads,
                    3 * self.head_dim)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer, value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_x_layer, 3)
        elif self.attention_head_type == 'groupedquery':
            # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
            mixed_x_layer, _ = self.query_key_value(x)
    
            # [sq, b, ((np + kvnp*2) * hn)] --> [sq, b, local_num_kv_heads, np//kvnp + 2, hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                (-1, self.n_local_heads // self.local_num_kv_heads + 2, self.head_dim)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, local_num_kv_heads, np//kvnp + 2, hn] --> 
            #   [sq, b, local_num_kv_heads, np//kvnp, hn], 2 * [sq, b, local_num_kv_heads, 1, hn]
            query_layer = mixed_x_layer[:,:,:,:-2]
            key_layer = mixed_x_layer[:,:,:,[-2]]
            value_layer = mixed_x_layer[:,:,:,[-1]]

        
        self.cache_k = self.cache_k.to(query_layer)
        self.cache_v = self.cache_v.to(query_layer)

        self.cache_k[start_pos : start_pos + seqlen, :bsz] = key_layer
        self.cache_v[start_pos : start_pos + seqlen, :bsz] = value_layer

        key_layer = self.cache_k[: start_pos + seqlen, :bsz]
        value_layer = self.cache_v[: start_pos + seqlen, :bsz]

        if self.attention_head_type == 'groupedquery':

            # TODO: now flash-attention only allowed multi-head, so we need copy kv_num_head to q_num_head
            sq, b, lkv, np_lkv, hn = query_layer.size()
            kv_size = [key_layer.size()[0], b, lkv, np_lkv, hn]
            key_layer = torch.broadcast_to(key_layer, kv_size)
            value_layer = torch.broadcast_to(value_layer, kv_size)
            query_layer, key_layer, value_layer  = [x.flatten(2, 3) for x in (query_layer, key_layer, value_layer)]

        query_layer, key_layer = RotaryEmbedding.apply_rotary(query_layer, key_layer, freqs_cis=freqs_cis)


        if not self.use_flash_attn:
            context_layer = self.core_attention(
                query_layer, key_layer, value_layer, attention_mask)
        else:
            q, k, v = [rearrange(x, 's b ... -> b s ...').contiguous()
                       for x in (query_layer, key_layer, value_layer)]
            if not self.sequence_parallel:
                with tensor_parallel.get_cuda_rng_tracker().fork():
                    context_layer = self.core_attention_flash(q, k, v)
            else:
                context_layer = self.core_attention_flash(q, k, v)
            context_layer = rearrange(context_layer, 'b s h d -> s b (h d)').contiguous()

        output, _ = self.wo(context_layer)
        return output


class LLaMAFeedForward(MegatronModule):
    def __init__(
        self,init_method, output_layer_init_method
    ):
        super().__init__()
        args = get_args()
        self.params_dtype = args.params_dtype

        if args.inner_hidden_dim is not None and isinstance(args.inner_hidden_dim, int) and args.inner_hidden_dim > 0:
            inn_hidden_dim = args.inner_hidden_dim
        else:
            ffn_expand_rate = 4
            inn_hidden_dim = int(2 * (args.hidden_size * ffn_expand_rate) / 3)

            # make SwiGLU hidden layer size multiple of large power of 2
            multiple_of = 256
            inn_hidden_dim = multiple_of * ((inn_hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = tensor_parallel.ColumnParallelLinear(
            args.hidden_size,
            inn_hidden_dim,
            params_dtype=self.params_dtype,
            gather_output=False,
            init_method=init_method,
            perform_initialization=False,
            use_cpu_initialization=True,
            skip_bias_add=True,
            bias=False,
        )
        
        self.w2 =  tensor_parallel.RowParallelLinear(
            inn_hidden_dim,
            args.hidden_size,
            params_dtype=self.params_dtype,
            input_is_parallel=True if args.tensor_model_parallel_size > 1 else False,
            init_method=output_layer_init_method,
            perform_initialization=False,
            use_cpu_initialization=True,
            skip_bias_add=True,
            bias=False,
        )

        self.w3 = tensor_parallel.ColumnParallelLinear(
            args.hidden_size,
            inn_hidden_dim,
            params_dtype=self.params_dtype,
            gather_output=False,
            init_method=init_method,
            perform_initialization=False,
            use_cpu_initialization=True,
            skip_bias_add=True,
            bias=False,
        )

    def forward(self, x):
        part_1, _ = self.w1(x)
        part_1 = F.silu(part_1)
        part_2, _ = self.w3(x)
        
        final, _ = self.w2(part_1 * part_2)
        return final


class LLaMATransformerBlock(MegatronModule):
    def __init__(self, init_method,
                 output_layer_init_method, layer_number):
        super().__init__()
        args = get_args()
        
        self.attention = LLaMAttention(
            init_method=init_method, output_layer_init_method=output_layer_init_method, layer_number=layer_number)
        self.feed_forward = LLaMAFeedForward(
            init_method, output_layer_init_method
        )
        self.layer_number = layer_number

        # epsilon: 1e-5
        self.attention_norm = RMSNorm(args.hidden_size, eps=args.layernorm_epsilon)
        self.ffn_norm = RMSNorm(args.hidden_size, eps=args.layernorm_epsilon)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class LLaMATransformer(MegatronModule):
    def __init__(self, init_method, output_layer_init_method):
        super().__init__()
        args = get_args()
        self.num_layers = args.num_layers
        self.seq_dimension = 0
        self.params_dtype = args.params_dtype

        self.tok_embeddings = tensor_parallel.VocabParallelEmbedding(
            args.padded_vocab_size, args.hidden_size,
            init_method=init_method,
            params_dtype=self.params_dtype,
            use_cpu_initialization=True,
            perform_initialization=False
        )
        
        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.num_layers):
            self.layers.append(LLaMATransformerBlock(init_method, output_layer_init_method, layer_id))

        self.norm = RMSNorm(args.hidden_size, eps=args.layernorm_epsilon)

        # mapping hidden_states to logits value
        self.output = tensor_parallel.ColumnParallelLinear(
            args.hidden_size,
            args.padded_vocab_size,
            params_dtype=self.params_dtype,
            gather_output=True,
            init_method=init_method,
            perform_initialization=False,
            use_cpu_initialization=True,
            skip_bias_add=True,
            bias=False
        )

        if args.rope_linear_scaling_factor > 1:
            self.rope = LinearScalingRotaryEmbedding(seq_dimension=self.seq_dimension, rope_theta=args.rope_theta, scaling_factor=args.rope_linear_scaling_factor)
        else:
            self.rope = RotaryEmbedding(seq_dimension=self.seq_dimension, rope_theta=args.rope_theta)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int, return_hidden=False) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = h.transpose(0, 1).contiguous()

        self.rope.set_devices_dtype(h)
        h_shape = h.shape
        h_shape = [s + start_pos if s_i ==0 else s for s_i, s in enumerate(h_shape)]
        freqs_cis = self.rope.get_freqs_cis(h_shape)

        attention_mask = None
        if seqlen > 1:
            attention_mask = torch.tril(torch.ones(
                (1, seqlen, seqlen), device=tokens.device)).view(
                    1, 1, seqlen, seqlen).type_as(h)
            attention_mask = (attention_mask < 0.5)
            
        hidden_list = []
        for layer_id, layer in enumerate(self.layers):

            # shape: [seq_len, bsz, hidden_size]
            h = layer(h, start_pos, freqs_cis, attention_mask)

            if return_hidden and layer_id in {0, int(self.num_layers/3), int(self.num_layers/5 * 4)}:
                hidden_list.append(h.float().transpose(0,1).contiguous())
        
        h = self.norm(h)
        h = h.transpose(0,1).contiguous()
        if return_hidden:
            hidden_list.append(h.float())

        output, _ = self.output(h)

        if return_hidden:
            return output.float(), hidden_list, self.output.weight.float()
        else:
            return output.float()