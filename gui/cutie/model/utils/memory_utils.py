import math
from typing import Optional, Union, Tuple

import torch

try:
    import triton  # noqa: F401

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


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
        values, indices = torch.topk(similarity, k=top_k, dim=1)
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
    values, indices = torch.topk(similarity, k=top_k, dim=1)
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
        values, order = torch.topk(values, k=top_k, dim=1)
        indices = indices.gather(1, order)
    return values, indices


def _chunked_topk_softmax_sparse(
    mk: torch.Tensor,
    ms: Optional[torch.Tensor],
    qk: torch.Tensor,
    qe: Optional[torch.Tensor],
    top_k: int,
    return_usage: bool = False,
) -> tuple:
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
        qe_float = qe.float()
        qk_weighted = (qk * qe).float()
        b_sq = (qe * qk.square()).float().sum(1, keepdim=False)
    else:
        qe_float = None
        qk_weighted = None
        b_sq = None

    topk_values = None
    topk_indices = None
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
        chunk_values, local_indices = torch.topk(sim_chunk, k=chunk_k, dim=1)
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

    maxes = topk_values.max(dim=1, keepdim=True).values
    topk_weights = torch.exp(topk_values - maxes)
    topk_weights /= torch.sum(topk_weights, dim=1, keepdim=True)

    if return_usage:
        usage = torch.zeros(bs, num_tokens, device=mk.device, dtype=topk_weights.dtype)
        usage.scatter_add_(1, topk_indices.reshape(bs, -1), topk_weights.reshape(bs, -1))
        return topk_weights, topk_indices, usage
    return topk_weights, topk_indices


def sparse_topk_affinity(
    mk: torch.Tensor,
    ms: Optional[torch.Tensor],
    qk: torch.Tensor,
    qe: Optional[torch.Tensor],
    top_k: int,
    *,
    return_usage: bool = False,
    backend: str = "auto",
) -> tuple:
    """Compute sparse top-k affinity without materializing the full similarity tensor.

    ``backend='triton'`` currently uses the same streamed PyTorch implementation
    as the CUDA fallback. The selection is kept explicit so a dedicated Triton
    kernel can be dropped in later without changing call sites.
    """
    resolved = _resolve_sparse_backend(backend, mk)

    if resolved == "pytorch" and not mk.is_cuda:
        similarity = get_similarity(mk, ms, qk, qe)
        return do_softmax_sparse(similarity, top_k=top_k, return_usage=return_usage)

    return _chunked_topk_softmax_sparse(
        mk, ms, qk, qe, top_k, return_usage=return_usage,
    )


def sparse_readout(
        v: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Memory readout using sparse top-k affinity (gather instead of dense BMM).

    Args:
        v: (bs, C, N) or (bs, num_objects, C, N) — memory values
        topk_indices: (bs, top_k, HW) — indices into N dimension
        topk_weights: (bs, top_k, HW) — normalised weights

    Returns:
        (bs, C, HW) or (bs, num_objects, C, HW)
    """
    multi_obj = v.dim() == 4
    if multi_obj:
        bs, num_objects, CV, N = v.shape
        v = v.reshape(bs, num_objects * CV, N)

    bs, C, N = v.shape
    top_k = topk_indices.shape[1]
    HW = topk_indices.shape[2]

    # Gather top-k values along N:  (bs, C, top_k*HW)
    idx = topk_indices.reshape(bs, 1, top_k * HW).expand(bs, C, -1)
    gathered = torch.gather(v, 2, idx).view(bs, C, top_k, HW)

    # Weighted sum over top_k dimension
    out = (gathered * topk_weights.unsqueeze(1)).sum(2)   # (bs, C, HW)

    if multi_obj:
        out = out.view(bs, num_objects, CV, HW)
    return out


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
