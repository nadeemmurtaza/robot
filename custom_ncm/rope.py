"""
Rotary Position Embedding (RoPE) for 1M+ Long Context Next-Concept Modeling.
Supports extreme sequence lengths (1,048,576 tokens / 1M) with:
1. Zero parameter / zero memory lookup tables (calculated on-the-fly)
2. High base frequency (theta = 10,000,000) for stable 1M context extrapolation
3. Compatible with TPU v5e-1 (bfloat16) and GPU/CPU
"""
import math
import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE for 1M Context Window (theta = 10,000,000).
    Extrapolates positional information without allocating giant tables.
    """
    def __init__(self, dim: int, max_seq_len: int = 1048576, base: float = 10000000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        # Cached inv_freq
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Calculates sinusoidal position embeddings for sequence length on the fly.
        x: [B, L, D]
        Returns: [1, L, D] position embeddings
        """
        device = x.device
        inv_freq = self.inv_freq.to(device)
        t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)  # [L, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [L, D]
        return emb.unsqueeze(0).to(x.dtype)
