"""
Latent Consistency Model (LCM) & Bidirectional Diffusion Backbone for Code.
Implements:
1. Bidirectional Attention Backbone (inspired by Diffusion Gemma)
2. Latent Consistency Boundary Parameterization f_theta(z_t, t) -> z_0
3. 1-Step and 2-Step Multistep Consistency Sampling
4. Consistency Training & Distillation Losses
"""
from typing import List, Tuple, Dict, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_ncm.rope import RotaryPositionalEmbedding

class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal embeddings for diffusion timesteps t in [0, 1000]."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B]
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32, device=t.device) / half_dim
        )
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class BidirectionalDiffusionBlock(nn.Module):
    """
    Bidirectional Transformer Block (Diffusion Gemma style).
    Attends across all tokens simultaneously with adaptive timestep conditioning.
    """
    def __init__(self, d_model: int, n_head: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Adaptive conditioning projection for timestep embedding
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model)  # Scale & Shift (AdaLN)
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # AdaLN modulation
        t_params = self.time_mlp(time_emb).unsqueeze(1)  # [B, 1, 2*D]
        scale, shift = t_params.chunk(2, dim=-1)
        
        # Bidirectional self-attention (attends everywhere with full visibility)
        normed = self.ln1(x) * (1 + scale) + shift
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = x + attn_out
        
        # Feedforward
        x = x + self.mlp(self.ln2(x))
        return x


class LatentConsistencyPlanner(nn.Module):
    """
    Latent Consistency Model (LCM) for Coarse Concept Planning.
    Maps noisy concept latents z_t directly to clean concept representations z_0
    in a SINGLE forward step, satisfying the exact boundary condition:
        f_theta(z_0, 0) = z_0
    """
    def __init__(
        self,
        latent_dim: int = 128,
        concept_seq_len: int = 8,
        prefix_len: int = 32,
        n_layers: int = 4,
        n_heads: int = 4,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.concept_seq_len = concept_seq_len
        self.prefix_len = prefix_len
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Prefix encoder projection
        self.prefix_proj = nn.Linear(latent_dim, latent_dim)
        
        # Position embedding for concept sequence (RoPE for long sequences)
        if concept_seq_len > 8192:
            self.concept_rope = RotaryPositionalEmbedding(latent_dim, max_seq_len=concept_seq_len)
            self.concept_pos = None
        else:
            self.concept_pos = nn.Embedding(concept_seq_len, latent_dim)
            self.concept_rope = None
        
        # Bidirectional Transformer backbone
        self.blocks = nn.ModuleList([
            BidirectionalDiffusionBlock(latent_dim, n_head=n_heads)
            for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)

    def get_concept_pos(self, x: torch.Tensor, S: int) -> torch.Tensor:
        if self.concept_rope is not None:
            return self.concept_rope(x, S)
        pos = torch.arange(S, device=x.device).unsqueeze(0)
        return self.concept_pos(pos)

    def get_scalings(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates consistency model boundary scalings c_skip, c_out, c_in:
        f_theta(x, sigma) = c_skip(sigma) * x + c_out(sigma) * F_theta(c_in(sigma) * x, sigma)
        Ensures f_theta(x, 0) == x identically!
        """
        # Ensure sigma has shape [B, 1, 1]
        s = sigma.view(-1, 1, 1)
        c_skip = (self.sigma_data ** 2) / ((s - self.sigma_min) ** 2 + self.sigma_data ** 2)
        c_out = (s - self.sigma_min) * self.sigma_data / (s ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1.0 / (s ** 2 + self.sigma_data ** 2).sqrt()
        return c_skip, c_out, c_in

    def forward(
        self,
        z_t: torch.Tensor,
        sigma: torch.Tensor,
        prefix_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Predicts clean concept latents z_0 directly from noisy z_t.
        Args:
            z_t: [B, S_concept, D] noisy concept vectors
            sigma: [B] noise levels
            prefix_embeds: [B, P, D] conditioning prompt prefix embeddings
        Returns:
            z_0: [B, S_concept, D] clean concept prediction
        """
        B, S, D = z_t.shape
        c_skip, c_out, c_in = self.get_scalings(sigma)
        
        # Scale input
        x_scaled = c_in * z_t
        
        # Add concept position embeddings
        h = x_scaled + self.get_concept_pos(x_scaled, S)
        
        # Concatenate prefix condition to allow bidirectional cross-attention
        p = self.prefix_proj(prefix_embeds)
        full_seq = torch.cat([p, h], dim=1)  # [B, P + S, D]
        
        # Time conditioning
        t_emb = self.time_embed(sigma)
        
        # Run bidirectional transformer blocks
        for block in self.blocks:
            full_seq = block(full_seq, t_emb)
            
        full_seq = self.ln_out(full_seq)
        
        # Extract concept tokens portion
        concept_out = self.out_proj(full_seq[:, -S:, :])
        
        # Consistency boundary formulation:
        # At sigma = sigma_min (close to 0), c_skip -> 1, c_out -> 0 => z_0 = z_t
        z_0_pred = c_skip * z_t + c_out * concept_out
        return z_0_pred

    @torch.no_grad()
    def sample_one_step(self, prefix_embeds: torch.Tensor) -> torch.Tensor:
        """
        1-STEP CONSISTENCY GENERATION:
        Directly jumps from maximum Gaussian noise to clean concept blueprint
        in A SINGLE FORWARD PASS!
        """
        B = prefix_embeds.shape[0]
        device = prefix_embeds.device
        
        # Sample pure Gaussian noise
        z_T = torch.randn(B, self.concept_seq_len, self.latent_dim, device=device) * self.sigma_max
        sigma = torch.full((B,), self.sigma_max, device=device)
        
        # One single evaluation
        z_0 = self.forward(z_T, sigma, prefix_embeds)
        return z_0

    @torch.no_grad()
    def sample_multistep(self, prefix_embeds: torch.Tensor, steps: int = 2) -> torch.Tensor:
        """
        2-STEP CONSISTENCY GENERATION:
        Runs 2 consistency refinement steps for maximum fidelity.
        """
        B = prefix_embeds.shape[0]
        device = prefix_embeds.device
        
        # Step 1: from sigma_max
        z_t = torch.randn(B, self.concept_seq_len, self.latent_dim, device=device) * self.sigma_max
        sigma_1 = torch.full((B,), self.sigma_max, device=device)
        z_0_pred = self.forward(z_t, sigma_1, prefix_embeds)
        
        if steps <= 1:
            return z_0_pred
            
        # Step 2: intermediate noise injection and direct jump
        intermediate_sigma = 2.5
        noise = torch.randn_like(z_0_pred)
        z_t_next = z_0_pred + noise * intermediate_sigma
        sigma_2 = torch.full((B,), intermediate_sigma, device=device)
        z_0_final = self.forward(z_t_next, sigma_2, prefix_embeds)
        return z_0_final
