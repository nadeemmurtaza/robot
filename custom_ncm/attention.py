"""
Memory-efficient attention for very long sequences (up to the 1M-token tier).

Plain dense attention needs O(L^2) memory for its score matrix. At L=1,048,576
that is roughly 1.1 trillion entries (~2.2 petabytes even in bf16) for ONE
layer, ONE head - no accelerator in existence holds that, TPU included.
WindowedAttention replaces dense attention with the standard Longformer/BigBird
recipe: every query attends to a handful of GLOBAL key positions (cheap, fixed
count) plus a LOCAL sliding window of nearby keys, measured by absolute
position within the full sequence. Memory drops to O(L * (window + global)).

This is an explicit, standard tradeoff for making 1M-token context possible at
all - a query far from the start of its key range no longer sees the WHOLE
key range, only the global anchor tokens and its own local neighborhood. For
short sequences (the common case at the small/medium/large tiers) this module
falls back to exact dense attention, so behaviour there is unchanged.
"""
import re
import torch
import torch.nn as nn
import torch.nn.functional as F


def convert_legacy_mha_state_dict(state_dict: dict) -> dict:
    """
    Checkpoints saved before WindowedAttention replaced nn.MultiheadAttention
    store one fused `<prefix>.in_proj_weight` ([3*D, D]) and `<prefix>.in_proj_bias`
    ([3*D]) per attention module (torch's packed Q/K/V projection). WindowedAttention
    instead has separate q_proj/k_proj/v_proj Linear layers. This splits the fused
    tensor into three equal chunks and renames them - mathematically identical
    weights, just unpacked - so old checkpoints keep working instead of silently
    losing their trained attention weights (out_proj is already named the same in
    both and needs no change).
    """
    converted = {}
    pattern = re.compile(r"^(.*\.attn)\.in_proj_(weight|bias)$")
    for key, value in state_dict.items():
        m = pattern.match(key)
        if not m:
            converted[key] = value
            continue
        prefix, kind = m.group(1), m.group(2)
        q, k, v = value.chunk(3, dim=0)
        converted[f"{prefix}.q_proj.{kind}"] = q
        converted[f"{prefix}.k_proj.{kind}"] = k
        converted[f"{prefix}.v_proj.{kind}"] = v
    return converted


class WindowedAttention(nn.Module):
    """
    Cross/self attention where `query` lives at absolute positions
    [q_start, q_start + Lq) of some full sequence, and `key`/`value` are a
    PREFIX of that same full sequence, i.e. positions [0, Lk). This matches
    exactly how the NCM block-causal structure and the VQ-VAE encoder/decoder
    use attention, so it's a drop-in replacement for nn.MultiheadAttention in
    both places.
    """

    def __init__(self, d_model: int, n_head: int, window_size: int = 2048,
                 num_global: int = 128, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.window_size = window_size
        self.num_global = num_global
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        return x.view(B, L, self.n_head, self.head_dim).transpose(1, 2)  # [B,H,L,hd]

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                q_start: int = 0) -> torch.Tensor:
        B, Lq, D = query.shape
        Lk = key.shape[1]

        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        # Exact dense attention whenever materializing it is cheap - keeps
        # small/medium/large tiers numerically identical to plain attention.
        if Lq * Lk <= 4 * self.window_size * max(Lq, 1):
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)
        else:
            out = self._windowed(q, k, v, q_start)

        out = out.transpose(1, 2).reshape(B, Lq, D)
        return self.out_proj(out)

    def _windowed(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  q_start: int) -> torch.Tensor:
        B, H, Lq, hd = q.shape
        Lk = k.shape[2]
        W = self.window_size
        G = min(self.num_global, Lk)

        # Global anchor keys: first G key positions, visible to every query.
        k_global, v_global = k[:, :, :G, :], v[:, :, :G, :]

        outputs = []
        chunk = max(W, 1)
        for start in range(0, Lq, chunk):
            end = min(start + chunk, Lq)
            q_chunk = q[:, :, start:end, :]
            abs_start = q_start + start
            abs_end = q_start + end - 1

            # Local window in absolute-position space; starts after the
            # global region so a key is never double-counted in softmax.
            lo = max(G, abs_start - W)
            hi = min(Lk, abs_end + W + 1)
            if lo < hi:
                k_local, v_local = k[:, :, lo:hi, :], v[:, :, lo:hi, :]
                k_cat = torch.cat([k_global, k_local], dim=2)
                v_cat = torch.cat([v_global, v_local], dim=2)
            else:
                k_cat, v_cat = k_global, v_global

            attn = torch.matmul(q_chunk, k_cat.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            outputs.append(torch.matmul(attn, v_cat))

        return torch.cat(outputs, dim=2)
