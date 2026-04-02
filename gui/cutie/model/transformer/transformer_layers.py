# Modified from PyTorch nn.Transformer

from typing import List, Callable

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from gui.cutie.model.channel_attn import CAResBlock


def _split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch_size, seq_len, dim = x.shape
    head_dim = dim // num_heads
    return x.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()


def _project_qkv(mha: nn.MultiheadAttention,
                 q: torch.Tensor,
                 k: torch.Tensor,
                 v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embed_dim = mha.embed_dim
    w_q, w_k, w_v = mha.in_proj_weight.split(embed_dim, dim=0)
    if mha.in_proj_bias is None:
        b_q = b_k = b_v = None
    else:
        b_q, b_k, b_v = mha.in_proj_bias.split(embed_dim, dim=0)

    q_proj = F.linear(q, w_q, b_q)
    k_proj = F.linear(k, w_k, b_k)
    v_proj = F.linear(v, w_v, b_v)
    return q_proj, k_proj, v_proj


def _bool_mask_to_float(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    float_mask = torch.zeros(mask.shape, device=mask.device, dtype=dtype)
    return float_mask.masked_fill(mask, float("-inf"))


def _prepare_sdpa_mask(attn_mask: torch.Tensor,
                       batch_size: int,
                       num_heads: int,
                       q_len: int,
                       kv_len: int,
                       dtype: torch.dtype,
                       device: torch.device,
                       key_padding_mask: torch.Tensor = None) -> torch.Tensor:
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask = _bool_mask_to_float(attn_mask, dtype)
        else:
            attn_mask = attn_mask.to(device=device, dtype=dtype)

        if attn_mask.dim() == 3:
            if attn_mask.shape[0] == batch_size * num_heads:
                attn_mask = attn_mask.view(batch_size, num_heads, q_len, kv_len)
            elif attn_mask.shape[0] == batch_size:
                attn_mask = attn_mask.view(batch_size, 1, q_len, kv_len)
            else:
                raise ValueError(
                    f"Unsupported attention mask shape {tuple(attn_mask.shape)}"
                )
        elif attn_mask.dim() == 2:
            attn_mask = attn_mask.view(1, 1, q_len, kv_len)
        elif attn_mask.dim() == 4:
            if attn_mask.shape[1] not in (1, num_heads):
                raise ValueError(
                    f"Unsupported attention mask shape {tuple(attn_mask.shape)}"
                )
        else:
            raise ValueError(f"Unsupported attention mask rank {attn_mask.dim()}")

    if key_padding_mask is not None:
        if key_padding_mask.dtype == torch.bool:
            key_padding_mask = _bool_mask_to_float(key_padding_mask, dtype)
        else:
            key_padding_mask = key_padding_mask.to(device=device, dtype=dtype)
        key_padding_mask = key_padding_mask.view(batch_size, 1, 1, kv_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = attn_mask + key_padding_mask

    return attn_mask


def _prepare_mha_mask(attn_mask: torch.Tensor,
                      batch_size: int,
                      num_heads: int,
                      q_len: int,
                      kv_len: int) -> torch.Tensor:
    if attn_mask is None:
        return None
    if attn_mask.dim() == 3 and attn_mask.shape[0] == batch_size:
        attn_mask = attn_mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
    if attn_mask.dim() == 4:
        if attn_mask.shape[1] == 1:
            attn_mask = attn_mask.expand(-1, num_heads, -1, -1)
        return attn_mask.reshape(batch_size * num_heads, q_len, kv_len)
    return attn_mask


def _sdpa_forward(mha: nn.MultiheadAttention,
                  q: torch.Tensor,
                  k: torch.Tensor,
                  v: torch.Tensor,
                  attn_mask: torch.Tensor = None,
                  key_padding_mask: torch.Tensor = None) -> torch.Tensor:
    q_proj, k_proj, v_proj = _project_qkv(mha, q, k, v)
    batch_size = q_proj.shape[0]
    q_len = q_proj.shape[1]
    kv_len = k_proj.shape[1]
    dtype = q_proj.dtype
    device = q_proj.device
    num_heads = mha.num_heads

    q_heads = _split_heads(q_proj, num_heads)
    k_heads = _split_heads(k_proj, num_heads)
    v_heads = _split_heads(v_proj, num_heads)
    attn_mask = _prepare_sdpa_mask(attn_mask,
                                   batch_size,
                                   num_heads,
                                   q_len,
                                   kv_len,
                                   dtype,
                                   device,
                                   key_padding_mask=key_padding_mask)
    out = F.scaled_dot_product_attention(
        q_heads,
        k_heads,
        v_heads,
        attn_mask=attn_mask,
        dropout_p=mha.dropout if mha.training else 0.0,
    )
    out = out.transpose(1, 2).contiguous().view(batch_size, q_len, mha.embed_dim)
    return F.linear(out, mha.out_proj.weight, mha.out_proj.bias)


class SelfAttention(nn.Module):
    def __init__(self,
                 dim: int,
                 nhead: int,
                 dropout: float = 0.0,
                 batch_first: bool = True,
                 add_pe_to_qkv: List[bool] = [True, True, False]):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=batch_first)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.add_pe_to_qkv = add_pe_to_qkv

    def forward(self,
                x: torch.Tensor,
                pe: torch.Tensor,
                attn_mask: torch.Tensor = None,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.norm(x)
        if any(self.add_pe_to_qkv):
            x_with_pe = x + pe
            q = x_with_pe if self.add_pe_to_qkv[0] else x
            k = x_with_pe if self.add_pe_to_qkv[1] else x
            v = x_with_pe if self.add_pe_to_qkv[2] else x
        else:
            q = k = v = x

        r = x
        x = _sdpa_forward(self.self_attn,
                          q,
                          k,
                          v,
                          attn_mask=attn_mask,
                          key_padding_mask=key_padding_mask)
        return r + self.dropout(x)


# https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention
class CrossAttention(nn.Module):
    def __init__(self,
                 dim: int,
                 nhead: int,
                 dropout: float = 0.0,
                 batch_first: bool = True,
                 add_pe_to_qkv: List[bool] = [True, True, False],
                 residual: bool = True,
                 norm: bool = True):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim,
                                                nhead,
                                                dropout=dropout,
                                                batch_first=batch_first)
        if norm:
            self.norm = nn.LayerNorm(dim)
        else:
            self.norm = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.add_pe_to_qkv = add_pe_to_qkv
        self.residual = residual

    def forward(self,
                x: torch.Tensor,
                mem: torch.Tensor,
                x_pe: torch.Tensor,
                mem_pe: torch.Tensor,
                attn_mask: torch.Tensor = None,
                *,
                need_weights: bool = False) -> (torch.Tensor, torch.Tensor):
        x = self.norm(x)
        if self.add_pe_to_qkv[0]:
            q = x + x_pe
        else:
            q = x

        if any(self.add_pe_to_qkv[1:]):
            mem_with_pe = mem + mem_pe
            k = mem_with_pe if self.add_pe_to_qkv[1] else mem
            v = mem_with_pe if self.add_pe_to_qkv[2] else mem
        else:
            k = v = mem
        r = x
        if need_weights:
            q_len = q.shape[1]
            kv_len = k.shape[1]
            x, weights = self.cross_attn(q,
                                         k,
                                         v,
                                         attn_mask=_prepare_mha_mask(attn_mask,
                                                                     q.shape[0],
                                                                     self.cross_attn.num_heads,
                                                                     q_len,
                                                                     kv_len),
                                         need_weights=True,
                                         average_attn_weights=False)
        else:
            x = _sdpa_forward(self.cross_attn, q, k, v, attn_mask=attn_mask)
            weights = None

        if self.residual:
            return r + self.dropout(x), weights
        else:
            return self.dropout(x), weights


class FFN(nn.Module):
    def __init__(self, dim_in: int, dim_ff: int, activation=F.relu):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, dim_ff)
        self.linear2 = nn.Linear(dim_ff, dim_in)
        self.norm = nn.LayerNorm(dim_in)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.norm(x)
        x = self.linear2(self.activation(self.linear1(x)))
        x = r + x
        return x


class PixelFFN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.conv = CAResBlock(dim, dim)

    def forward(self, pixel: torch.Tensor, pixel_flat: torch.Tensor) -> torch.Tensor:
        # pixel: batch_size * num_objects * dim * H * W
        # pixel_flat: (batch_size*num_objects) * (H*W) * dim
        bs, num_objects, _, h, w = pixel.shape
        pixel_flat = pixel_flat.view(bs * num_objects, h, w, self.dim)
        pixel_flat = pixel_flat.permute(0, 3, 1, 2)

        x = self.conv(pixel_flat)
        x = x.reshape(bs, num_objects, self.dim, h, w)
        return x


class OutputFFN(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, activation=F.relu):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, dim_out)
        self.linear2 = nn.Linear(dim_out, dim_out)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.activation(self.linear1(x)))
        return x


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
