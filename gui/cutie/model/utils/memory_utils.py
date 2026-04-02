import math
from contextlib import nullcontext
from typing import Optional, Union, Tuple

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


_MAX_TRITON_TOPK = 64


if _TRITON_AVAILABLE:

    _AFFINITY_TRITON_CONFIGS = [
        triton.Config({'BLOCK_N': 128, 'BLOCK_HW': 8, 'BLOCK_CK': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_HW': 16, 'BLOCK_CK': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_HW': 32, 'BLOCK_CK': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_HW': 8, 'BLOCK_CK': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_HW': 16, 'BLOCK_CK': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_HW': 32, 'BLOCK_CK': 64}, num_warps=8, num_stages=3),
    ]

    @triton.jit
    def _select_topk_rows(
        values,
        indices,
        BLOCK_ROWS: tl.constexpr,
        CANDIDATES: tl.constexpr,
        TOPK_PAD: tl.constexpr,
    ):
        # Encode each score together with its token id so Triton's bitonic top-k
        # can select both in one pass along the minor dimension.
        low_mask = 0xFFFFFFFF
        value_bits = values.to(tl.int32, bitcast=True)
        ordered_bits = value_bits ^ ((value_bits >> 31) & 0x7FFFFFFF)
        tie_break = low_mask - indices.to(tl.int64)
        packed = (ordered_bits.to(tl.int64) << 32) | tie_break
        top_packed = tl.topk(packed, k=TOPK_PAD, dim=1)

        top_ordered_bits = (top_packed >> 32).to(tl.int32)
        restored_bits = top_ordered_bits ^ ((top_ordered_bits >> 31) & 0x7FFFFFFF)
        out_values = restored_bits.to(tl.float32, bitcast=True)
        out_indices = (low_mask - (top_packed & low_mask)).to(tl.int32)
        return out_values, out_indices


    @triton.autotune(configs=_AFFINITY_TRITON_CONFIGS, key=["hw", "ck"])
    @triton.jit
    def _sparse_topk_affinity_qe_kernel(
        mk_ptr,
        ms_ptr,
        qe_ptr,
        qk_weighted_ptr,
        b_sq_ptr,
        values_ptr,
        indices_ptr,
        stride_mk_b,
        stride_mk_c,
        stride_mk_n,
        stride_ms_b,
        stride_ms_n,
        stride_qe_b,
        stride_qe_c,
        stride_qe_hw,
        stride_qkw_b,
        stride_qkw_c,
        stride_qkw_hw,
        stride_bsq_b,
        stride_bsq_hw,
        stride_out_b,
        stride_out_block,
        stride_out_k,
        stride_out_hw,
        stride_idx_b,
        stride_idx_block,
        stride_idx_k,
        stride_idx_hw,
        num_tokens,
        hw,
        ck,
        scale,
        HAS_MS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_HW: tl.constexpr,
        BLOCK_CK: tl.constexpr,
        TOPK_PAD: tl.constexpr,
    ):
        pid_hw = tl.program_id(0)
        pid_block = tl.program_id(1)
        pid_b = tl.program_id(2)

        hw_offsets = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offsets < hw
        token_offsets = pid_block * BLOCK_N + tl.arange(0, BLOCK_N)
        token_mask = token_offsets < num_tokens

        b_sq = tl.load(
            b_sq_ptr + pid_b * stride_bsq_b + hw_offsets * stride_bsq_hw,
            mask=hw_mask,
            other=0.0,
        ).to(tl.float32)

        sim = tl.zeros((BLOCK_HW, BLOCK_N), tl.float32)

        for ck_start in tl.range(0, ck, BLOCK_CK):
            c_offsets = ck_start + tl.arange(0, BLOCK_CK)
            c_mask = c_offsets < ck

            mk = tl.load(
                mk_ptr
                + pid_b * stride_mk_b
                + c_offsets[:, None] * stride_mk_c
                + token_offsets[None, :] * stride_mk_n,
                mask=c_mask[:, None] & token_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            mk_sq = mk * mk

            qe = tl.load(
                qe_ptr
                + pid_b * stride_qe_b
                + c_offsets[:, None] * stride_qe_c
                + hw_offsets[None, :] * stride_qe_hw,
                mask=c_mask[:, None] & hw_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            qk_weighted = tl.load(
                qk_weighted_ptr
                + pid_b * stride_qkw_b
                + c_offsets[:, None] * stride_qkw_c
                + hw_offsets[None, :] * stride_qkw_hw,
                mask=c_mask[:, None] & hw_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            sim += 2.0 * tl.dot(tl.trans(qk_weighted), mk, input_precision="ieee")
            sim -= tl.dot(tl.trans(qe), mk_sq, input_precision="ieee")

        sim -= b_sq[:, None]

        if HAS_MS:
            ms = tl.load(
                ms_ptr + pid_b * stride_ms_b + token_offsets * stride_ms_n,
                mask=token_mask,
                other=0.0,
            ).to(tl.float32)
            sim *= ms[None, :]

        sim *= scale
        sim = tl.where(hw_mask[:, None] & token_mask[None, :], sim, -float("inf"))

        token_ids = tl.broadcast_to(token_offsets[None, :].to(tl.int32), (BLOCK_HW, BLOCK_N))
        local_values, local_indices = _select_topk_rows(
            sim,
            token_ids,
            BLOCK_ROWS=BLOCK_HW,
            CANDIDATES=BLOCK_N,
            TOPK_PAD=TOPK_PAD,
        )

        rank_offsets = tl.arange(0, TOPK_PAD)
        tl.store(
            values_ptr
            + pid_b * stride_out_b
            + hw_offsets[:, None] * stride_out_hw
            + pid_block * stride_out_block
            + rank_offsets[None, :] * stride_out_k,
            local_values,
            mask=hw_mask[:, None],
        )
        tl.store(
            indices_ptr
            + pid_b * stride_idx_b
            + hw_offsets[:, None] * stride_idx_hw
            + pid_block * stride_idx_block
            + rank_offsets[None, :] * stride_idx_k,
            local_indices,
            mask=hw_mask[:, None],
        )


    @triton.autotune(configs=_AFFINITY_TRITON_CONFIGS, key=["hw", "ck"])
    @triton.jit
    def _sparse_topk_affinity_no_qe_kernel(
        mk_ptr,
        ms_ptr,
        qk_ptr,
        values_ptr,
        indices_ptr,
        stride_mk_b,
        stride_mk_c,
        stride_mk_n,
        stride_ms_b,
        stride_ms_n,
        stride_qk_b,
        stride_qk_c,
        stride_qk_hw,
        stride_out_b,
        stride_out_block,
        stride_out_k,
        stride_out_hw,
        stride_idx_b,
        stride_idx_block,
        stride_idx_k,
        stride_idx_hw,
        num_tokens,
        hw,
        ck,
        scale,
        HAS_MS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_HW: tl.constexpr,
        BLOCK_CK: tl.constexpr,
        TOPK_PAD: tl.constexpr,
    ):
        pid_hw = tl.program_id(0)
        pid_block = tl.program_id(1)
        pid_b = tl.program_id(2)

        hw_offsets = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offsets < hw
        token_offsets = pid_block * BLOCK_N + tl.arange(0, BLOCK_N)
        token_mask = token_offsets < num_tokens

        sim = tl.zeros((BLOCK_HW, BLOCK_N), tl.float32)
        mk_sq = tl.zeros((BLOCK_N,), tl.float32)

        for ck_start in tl.range(0, ck, BLOCK_CK):
            c_offsets = ck_start + tl.arange(0, BLOCK_CK)
            c_mask = c_offsets < ck

            mk = tl.load(
                mk_ptr
                + pid_b * stride_mk_b
                + c_offsets[:, None] * stride_mk_c
                + token_offsets[None, :] * stride_mk_n,
                mask=c_mask[:, None] & token_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            qk = tl.load(
                qk_ptr
                + pid_b * stride_qk_b
                + c_offsets[:, None] * stride_qk_c
                + hw_offsets[None, :] * stride_qk_hw,
                mask=c_mask[:, None] & hw_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            sim += 2.0 * tl.dot(tl.trans(qk), mk, input_precision="ieee")
            mk_sq += tl.sum(mk * mk, axis=0)

        sim -= mk_sq[None, :]

        if HAS_MS:
            ms = tl.load(
                ms_ptr + pid_b * stride_ms_b + token_offsets * stride_ms_n,
                mask=token_mask,
                other=0.0,
            ).to(tl.float32)
            sim *= ms[None, :]

        sim *= scale
        sim = tl.where(hw_mask[:, None] & token_mask[None, :], sim, -float("inf"))

        token_ids = tl.broadcast_to(token_offsets[None, :].to(tl.int32), (BLOCK_HW, BLOCK_N))
        local_values, local_indices = _select_topk_rows(
            sim,
            token_ids,
            BLOCK_ROWS=BLOCK_HW,
            CANDIDATES=BLOCK_N,
            TOPK_PAD=TOPK_PAD,
        )

        rank_offsets = tl.arange(0, TOPK_PAD)
        tl.store(
            values_ptr
            + pid_b * stride_out_b
            + hw_offsets[:, None] * stride_out_hw
            + pid_block * stride_out_block
            + rank_offsets[None, :] * stride_out_k,
            local_values,
            mask=hw_mask[:, None],
        )
        tl.store(
            indices_ptr
            + pid_b * stride_idx_b
            + hw_offsets[:, None] * stride_idx_hw
            + pid_block * stride_idx_block
            + rank_offsets[None, :] * stride_idx_k,
            local_indices,
            mask=hw_mask[:, None],
        )


    @triton.jit
    def _sparse_readout_kernel(
        value_ptr,
        indices_ptr,
        weights_ptr,
        out_ptr,
        stride_value_b,
        stride_value_n,
        stride_value_c,
        stride_idx_b,
        stride_idx_k,
        stride_idx_hw,
        stride_weight_b,
        stride_weight_k,
        stride_weight_hw,
        stride_out_b,
        stride_out_c,
        stride_out_hw,
        num_tokens,
        hw,
        total_channels,
        TOPK: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        pid_hw = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_b = tl.program_id(2)

        hw_offsets = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
        c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

        hw_mask = hw_offsets < hw
        c_mask = c_offsets < total_channels

        acc = tl.zeros((BLOCK_C, BLOCK_HW), tl.float32)

        for rank in tl.static_range(TOPK):
            token_indices = tl.load(
                indices_ptr + pid_b * stride_idx_b + rank * stride_idx_k + hw_offsets * stride_idx_hw,
                mask=hw_mask,
                other=0,
            ).to(tl.int32)
            weights = tl.load(
                weights_ptr
                + pid_b * stride_weight_b
                + rank * stride_weight_k
                + hw_offsets * stride_weight_hw,
                mask=hw_mask,
                other=0.0,
            ).to(tl.float32)

            values = tl.load(
                value_ptr
                + pid_b * stride_value_b
                + token_indices[None, :] * stride_value_n
                + c_offsets[:, None] * stride_value_c,
                mask=c_mask[:, None] & hw_mask[None, :] & (token_indices[None, :] < num_tokens),
                other=0.0,
            ).to(tl.float32)
            acc += values * weights[None, :]

        tl.store(
            out_ptr
            + pid_b * stride_out_b
            + c_offsets[:, None] * stride_out_c
            + hw_offsets[None, :] * stride_out_hw,
            acc,
            mask=c_mask[:, None] & hw_mask[None, :],
        )


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def _choose_triton_topk_pad(top_k: int) -> Optional[int]:
    topk_pad = _next_power_of_two(top_k)
    if topk_pad > _MAX_TRITON_TOPK:
        return None
    return topk_pad


def _choose_triton_candidate_dtype(tensor: torch.Tensor) -> torch.dtype:
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return tensor.dtype
    return torch.float32


# @torch.jit.script
def get_similarity(mk: torch.Tensor,
                   ms: torch.Tensor,
                   qk: torch.Tensor,
                   qe: torch.Tensor,
                   add_batch_dim: bool = False) -> torch.Tensor:
    # used for training/inference and memory reading/memory potentiation
    # mk: B x CK x [N]    - Memory keys
    # ms: B x  1 x [N]    - Memory shrinkage
    # qk: B x CK x [HW/P] - Query keys
    # qe: B x CK x [HW/P] - Query selection
    # Dimensions in [] are flattened
    if add_batch_dim:
        mk, ms = mk.unsqueeze(0), ms.unsqueeze(0)
        qk, qe = qk.unsqueeze(0), qe.unsqueeze(0)

    CK = mk.shape[1]
    mk = mk.flatten(start_dim=2)
    ms = ms.flatten(start_dim=1).unsqueeze(2) if ms is not None else None
    qk = qk.flatten(start_dim=2)
    qe = qe.flatten(start_dim=2) if qe is not None else None

    if qe is not None:
        # See XMem's appendix for derivation
        mk = mk.transpose(1, 2)
        a_sq = (mk.pow(2) @ qe)
        two_ab = 2 * (mk @ (qk * qe))
        b_sq = (qe * qk.pow(2)).sum(1, keepdim=True)
        similarity = (-a_sq + two_ab - b_sq)
    else:
        # similar to STCN if we don't have the selection term
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        two_ab = 2 * (mk.transpose(1, 2) @ qk)
        similarity = (-a_sq + two_ab)

    if ms is not None:
        similarity = similarity * ms / math.sqrt(CK)  # B*N*HW
    else:
        similarity = similarity / math.sqrt(CK)  # B*N*HW

    return similarity


def do_softmax(
        similarity: torch.Tensor,
        top_k: Optional[int] = None,
        inplace: bool = False,
        return_usage: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    # normalize similarity with top-k softmax
    # similarity: B x N x [HW/P]
    # use inplace with care
    if top_k is not None:
        values, indices = torch.topk(similarity, k=top_k, dim=1, sorted=False)
        maxes = values.max(dim=1, keepdim=True).values
        x_exp = torch.exp(values - maxes)
        x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
        if inplace:
            similarity.zero_().scatter_(1, indices, x_exp)  # B*N*HW
            affinity = similarity
        else:
            affinity = torch.zeros_like(similarity).scatter_(1, indices, x_exp)  # B*N*HW
    else:
        maxes = torch.max(similarity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(similarity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum
        indices = None

    if return_usage:
        return affinity, affinity.sum(dim=2)

    return affinity


def do_softmax_sparse(
        similarity: torch.Tensor,
        top_k: int,
        return_usage: bool = False,
) -> tuple:
    """Top-k softmax returning sparse format for efficient gather-based readout.

    Instead of scattering weights into a dense (N × HW) matrix and doing a full
    BMM (99.7% multiply-by-zero when top_k=30, N=10000), this keeps only the
    top-k indices and weights so ``sparse_readout`` can gather just what it needs.

    Args:
        similarity: (bs, N, HW)
        top_k: number of entries to keep per spatial position

    Returns:
        topk_weights: (bs, top_k, HW) — normalised softmax weights
        topk_indices: (bs, top_k, HW) — indices into N dimension
        usage (optional): (bs, N) — per-token sum of weights across HW
    """
    values, indices = torch.topk(similarity, k=top_k, dim=1, sorted=False)
    maxes = values.max(dim=1, keepdim=True).values
    x_exp = torch.exp(values - maxes)
    x_exp /= torch.sum(x_exp, dim=1, keepdim=True)

    if return_usage:
        bs, N = similarity.shape[0], similarity.shape[1]
        usage = torch.zeros(bs, N, device=similarity.device, dtype=x_exp.dtype)
        usage.scatter_add_(1, indices.reshape(bs, -1), x_exp.reshape(bs, -1))
        return x_exp, indices, usage

    return x_exp, indices


def _resolve_sparse_backend(backend: str, tensor: torch.Tensor) -> str:
    if backend not in {"auto", "pytorch", "triton"}:
        raise ValueError(f"Unsupported readout backend '{backend}'")
    if backend == "auto":
        if tensor.is_cuda and _TRITON_AVAILABLE:
            return "triton"
        return "pytorch"
    return backend


def _choose_chunk_size(num_tokens: int, hw: int, top_k: int, device: torch.device) -> int:
    if num_tokens <= top_k:
        return num_tokens
    if device.type == "cuda":
        target_elements = 8 * 1024 * 1024
    else:
        target_elements = 1 * 1024 * 1024
    chunk = max(top_k, target_elements // max(1, hw))
    return min(num_tokens, chunk)


def _merge_topk(
    curr_values: Optional[torch.Tensor],
    curr_indices: Optional[torch.Tensor],
    new_values: torch.Tensor,
    new_indices: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if curr_values is None:
        values = new_values
        indices = new_indices
    else:
        values = torch.cat([curr_values, new_values], dim=1)
        indices = torch.cat([curr_indices, new_indices], dim=1)

    if values.shape[1] > top_k:
        values, order = torch.topk(values, k=top_k, dim=1, sorted=False)
        indices = indices.gather(1, order)
    return values, indices


def _chunked_topk_softmax_sparse(
    mk: torch.Tensor,
    ms: Optional[torch.Tensor],
    qk: torch.Tensor,
    qe: Optional[torch.Tensor],
    top_k: int,
    profiler=None,
    return_usage: bool = False,
) -> tuple:
    section = profiler.section if profiler is not None else nullcontext
    mk = mk.flatten(start_dim=2)
    qk = qk.flatten(start_dim=2)
    qe = qe.flatten(start_dim=2) if qe is not None else None
    ms = ms.flatten(start_dim=1).unsqueeze(2) if ms is not None else None

    bs, ck, num_tokens = mk.shape
    hw = qk.shape[2]
    top_k = min(top_k, num_tokens)
    scale = 1.0 / math.sqrt(ck)
    chunk_size = _choose_chunk_size(num_tokens, hw, top_k, mk.device)

    qk_float = qk.float()
    if qe is not None:
        with section('affinity_query_precompute'):
            qe_float = qe.float()
            qk_weighted = (qk * qe).float()
            b_sq = (qe * qk.square()).float().sum(1, keepdim=False)
    else:
        qe_float = None
        qk_weighted = None
        b_sq = None

    topk_values = None
    topk_indices = None
    with section('affinity_candidate_build'):
        for start in range(0, num_tokens, chunk_size):
            end = min(start + chunk_size, num_tokens)
            mk_chunk = mk[:, :, start:end].float()
            if qe_float is not None:
                mk_t = mk_chunk.transpose(1, 2)
                sim_chunk = -(mk_t.square() @ qe_float)
                sim_chunk += 2 * (mk_t @ qk_weighted)
                sim_chunk -= b_sq.unsqueeze(1)
            else:
                mk_t = mk_chunk.transpose(1, 2)
                sim_chunk = -mk_chunk.square().sum(1, keepdim=False).unsqueeze(2)
                sim_chunk = sim_chunk + 2 * (mk_t @ qk_float)

            if ms is not None:
                sim_chunk *= ms[:, start:end].float()
            sim_chunk *= scale

            chunk_k = min(top_k, end - start)
            chunk_values, local_indices = torch.topk(sim_chunk, k=chunk_k, dim=1, sorted=False)
            token_indices = torch.arange(start,
                                         end,
                                         device=mk.device,
                                         dtype=torch.long).view(1, -1, 1).expand(bs, -1, hw)
            chunk_indices = token_indices.gather(1, local_indices)
            topk_values, topk_indices = _merge_topk(topk_values,
                                                    topk_indices,
                                                    chunk_values,
                                                    chunk_indices,
                                                    top_k)

    with section('affinity_softmax'):
        maxes = topk_values.max(dim=1, keepdim=True).values
        topk_weights = torch.exp(topk_values - maxes)
        topk_weights /= torch.sum(topk_weights, dim=1, keepdim=True)

    if return_usage:
        with section('affinity_usage'):
            usage = torch.zeros(bs, num_tokens, device=mk.device, dtype=topk_weights.dtype)
            usage.scatter_add_(1, topk_indices.reshape(bs, -1), topk_weights.reshape(bs, -1))
        return topk_weights, topk_indices, usage
    return topk_weights, topk_indices


def _triton_topk_affinity(
    mk: torch.Tensor,
    ms: Optional[torch.Tensor],
    qk: torch.Tensor,
    qe: Optional[torch.Tensor],
    top_k: int,
    profiler=None,
    return_usage: bool = False,
) -> tuple:
    section = profiler.section if profiler is not None else nullcontext
    mk = mk.flatten(start_dim=2).contiguous()
    qk = qk.flatten(start_dim=2).contiguous()
    qe = qe.flatten(start_dim=2).contiguous() if qe is not None else None
    ms = ms.flatten(start_dim=1).contiguous() if ms is not None else None

    bs, ck, num_tokens = mk.shape
    hw = qk.shape[2]
    top_k = min(top_k, num_tokens)
    topk_pad = _choose_triton_topk_pad(top_k)
    if topk_pad is None:
        return _chunked_topk_softmax_sparse(mk, ms, qk, qe, top_k, return_usage=return_usage)

    block_n = 128
    num_blocks = triton.cdiv(num_tokens, block_n)
    scale = 1.0 / math.sqrt(ck)
    candidate_dtype = _choose_triton_candidate_dtype(mk)
    block_values = torch.empty((bs, hw, num_blocks, topk_pad),
                               device=mk.device,
                               dtype=candidate_dtype)
    block_indices = torch.empty((bs, hw, num_blocks, topk_pad), device=mk.device, dtype=torch.int32)
    grid = lambda meta: (triton.cdiv(hw, meta["BLOCK_HW"]), num_blocks, bs)

    with section('affinity_candidate_build'):
        if qe is not None:
            with section('affinity_query_precompute'):
                qk_weighted = (qk * qe).contiguous()
                b_sq = (qe * qk.square()).sum(1).contiguous()

            with section('affinity_triton_kernel'):
                _sparse_topk_affinity_qe_kernel[grid](
                    mk,
                    ms if ms is not None else mk.new_empty((bs, 1)),
                    qe,
                    qk_weighted,
                    b_sq,
                    block_values,
                    block_indices,
                    mk.stride(0),
                    mk.stride(1),
                    mk.stride(2),
                    ms.stride(0) if ms is not None else 0,
                    ms.stride(1) if ms is not None else 0,
                    qe.stride(0),
                    qe.stride(1),
                    qe.stride(2),
                    qk_weighted.stride(0),
                    qk_weighted.stride(1),
                    qk_weighted.stride(2),
                    b_sq.stride(0),
                    b_sq.stride(1),
                    block_values.stride(0),
                    block_values.stride(2),
                    block_values.stride(3),
                    block_values.stride(1),
                    block_indices.stride(0),
                    block_indices.stride(2),
                    block_indices.stride(3),
                    block_indices.stride(1),
                    num_tokens,
                    hw,
                    ck,
                    scale,
                    HAS_MS=ms is not None,
                    TOPK_PAD=topk_pad,
                )
        else:
            with section('affinity_triton_kernel'):
                _sparse_topk_affinity_no_qe_kernel[grid](
                    mk,
                    ms if ms is not None else mk.new_empty((bs, 1)),
                    qk,
                    block_values,
                    block_indices,
                    mk.stride(0),
                    mk.stride(1),
                    mk.stride(2),
                    ms.stride(0) if ms is not None else 0,
                    ms.stride(1) if ms is not None else 0,
                    qk.stride(0),
                    qk.stride(1),
                    qk.stride(2),
                    block_values.stride(0),
                    block_values.stride(2),
                    block_values.stride(3),
                    block_values.stride(1),
                    block_indices.stride(0),
                    block_indices.stride(2),
                    block_indices.stride(3),
                    block_indices.stride(1),
                    num_tokens,
                    hw,
                    ck,
                    scale,
                    HAS_MS=ms is not None,
                    TOPK_PAD=topk_pad,
                )

    with section('affinity_select_topk'):
        candidate_values = block_values.reshape(bs, hw, num_blocks * topk_pad)
        candidate_indices = block_indices.reshape(bs, hw, num_blocks * topk_pad)
        topk_values_hw, order = torch.topk(candidate_values, k=top_k, dim=2, sorted=False)
        indices_hw = candidate_indices.gather(2, order)

    with section('affinity_softmax'):
        topk_values = topk_values_hw.transpose(1, 2).contiguous().float()
        indices = indices_hw.transpose(1, 2).contiguous().to(torch.long)
        maxes = topk_values.max(dim=1, keepdim=True).values
        weights = torch.exp(topk_values - maxes)
        weights /= torch.sum(weights, dim=1, keepdim=True)

    if return_usage:
        with section('affinity_usage'):
            usage = torch.zeros(bs, num_tokens, device=mk.device, dtype=weights.dtype)
            usage.scatter_add_(1, indices.reshape(bs, -1), weights.reshape(bs, -1))
        return weights, indices, usage
    return weights, indices


def sparse_topk_affinity(
    mk: torch.Tensor,
    ms: Optional[torch.Tensor],
    qk: torch.Tensor,
    qe: Optional[torch.Tensor],
    top_k: int,
    *,
    profiler=None,
    return_usage: bool = False,
    backend: str = "auto",
) -> tuple:
    """Compute sparse top-k affinity without materializing the full similarity tensor."""
    resolved = _resolve_sparse_backend(backend, mk)

    if resolved == "triton" and mk.is_cuda and _TRITON_AVAILABLE:
        return _triton_topk_affinity(
            mk, ms, qk, qe, top_k, profiler=profiler, return_usage=return_usage)

    if not mk.is_cuda:
        similarity = get_similarity(mk, ms, qk, qe)
        return do_softmax_sparse(similarity, top_k=top_k, return_usage=return_usage)

    return _chunked_topk_softmax_sparse(
        mk, ms, qk, qe, top_k, profiler=profiler, return_usage=return_usage,
    )


def _prepare_token_major_values(
    v: Optional[torch.Tensor],
    v_token_major: Optional[torch.Tensor],
) -> tuple[torch.Tensor, bool, Optional[tuple[int, int]]]:
    if v_token_major is None:
        if v is None:
            raise ValueError("sparse_readout requires either v or v_token_major")
        multi_obj = v.dim() == 4
        if multi_obj:
            bs, num_objects, cv, _ = v.shape
            value_tm = v.permute(0, 3, 1, 2).reshape(bs, v.shape[-1], num_objects * cv).contiguous()
            return value_tm, True, (num_objects, cv)
        return v.transpose(1, 2).contiguous(), False, None

    if v_token_major.dim() == 4:
        bs, num_objects, _, cv = v_token_major.shape
        value_tm = v_token_major.permute(0, 2, 1, 3).reshape(bs, v_token_major.shape[2],
                                                              num_objects * cv).contiguous()
        return value_tm, True, (num_objects, cv)

    return v_token_major.contiguous(), False, None


def _sparse_readout_pytorch(
    v: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    multi_obj = v.dim() == 4
    if multi_obj:
        bs, num_objects, cv, n = v.shape
        v = v.reshape(bs, num_objects * cv, n)

    bs, c, _ = v.shape
    top_k = topk_indices.shape[1]
    hw = topk_indices.shape[2]

    idx = topk_indices.reshape(bs, 1, top_k * hw).expand(bs, c, -1)
    gathered = torch.gather(v, 2, idx).view(bs, c, top_k, hw)
    out = (gathered * topk_weights.unsqueeze(1)).sum(2)

    if multi_obj:
        out = out.view(bs, num_objects, cv, hw)
    return out


def _triton_sparse_readout(
    v: Optional[torch.Tensor],
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    v_token_major: Optional[torch.Tensor],
) -> torch.Tensor:
    value_tm, multi_obj, multi_shape = _prepare_token_major_values(v, v_token_major)
    bs, num_tokens, total_channels = value_tm.shape
    top_k = topk_indices.shape[1]
    hw = topk_indices.shape[2]

    out = torch.empty((bs, total_channels, hw), device=value_tm.device, dtype=torch.float32)
    index_i32 = topk_indices.contiguous().to(torch.int32)
    weights = topk_weights.contiguous()

    block_hw = 16
    block_c = 32
    grid = (triton.cdiv(hw, block_hw), triton.cdiv(total_channels, block_c), bs)
    _sparse_readout_kernel[grid](
        value_tm,
        index_i32,
        weights,
        out,
        value_tm.stride(0),
        value_tm.stride(1),
        value_tm.stride(2),
        index_i32.stride(0),
        index_i32.stride(1),
        index_i32.stride(2),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_tokens,
        hw,
        total_channels,
        TOPK=top_k,
        BLOCK_C=block_c,
        BLOCK_HW=block_hw,
        num_warps=4,
    )

    if multi_obj:
        num_objects, cv = multi_shape
        out = out.view(bs, num_objects, cv, hw)
    return out


def sparse_readout(
        v: Optional[torch.Tensor],
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        backend: str = "auto",
        v_token_major: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Memory readout using sparse top-k affinity (gather instead of dense BMM).

    Args:
        v: (bs, C, N) or (bs, num_objects, C, N) — memory values. Can be ``None``
            when ``v_token_major`` is provided and the Triton backend is used.
        topk_indices: (bs, top_k, HW) — indices into N dimension
        topk_weights: (bs, top_k, HW) — normalised weights
        backend: auto | pytorch | triton
        v_token_major: optional token-major cache with shape (bs, N, C) or
            (bs, num_objects, N, C)

    Returns:
        (bs, C, HW) or (bs, num_objects, C, HW)
    """
    resolved = _resolve_sparse_backend(backend, topk_weights)
    if resolved == "triton" and topk_weights.is_cuda and _TRITON_AVAILABLE:
        return _triton_sparse_readout(v, topk_indices, topk_weights, v_token_major)
    if v is None:
        raise ValueError("sparse_readout requires v for non-Triton backends")
    return _sparse_readout_pytorch(v, topk_indices, topk_weights)


def get_affinity(mk: torch.Tensor, ms: torch.Tensor, qk: torch.Tensor,
                 qe: torch.Tensor) -> torch.Tensor:
    # shorthand used in training with no top-k
    similarity = get_similarity(mk, ms, qk, qe)
    affinity = do_softmax(similarity)
    return affinity


def readout(affinity: torch.Tensor, mv: torch.Tensor) -> torch.Tensor:
    B, CV, T, H, W = mv.shape

    mo = mv.view(B, CV, T * H * W)
    mem = torch.bmm(mo, affinity)
    mem = mem.view(B, CV, H, W)

    return mem
