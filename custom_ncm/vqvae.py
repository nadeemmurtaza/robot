"""
Multiscale Residual Vector Quantizer (VQ-VAE) for Next-Concept Modeling.
Encodes code tokens into a hierarchy of coarse-to-fine discrete concept codes.
"""
from typing import List, Tuple, Dict, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class VectorQuantizer(nn.Module):
    """
    Standard Vector Quantizer with Straight-Through Estimator and Commitment Loss.
    """
    def __init__(self, codebook_size: int, embed_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(codebook_size, embed_dim)
        # Uniform initialization scaled by 1 / codebook_size
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: [B, S, D] continuous vectors to quantize
        Returns:
            z_q: [B, S, D] quantized vectors with straight-through gradient
            loss: scalar VQ commitment loss
            indices: [B, S] discrete codebook indices
        """
        B, S, D = z.shape
        flat_z = z.reshape(-1, D)
        
        # Distance ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z.e
        distances = (
            torch.sum(flat_z ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_z, self.embedding.weight.t())
        )
        
        indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(indices).view(B, S, D)
        indices = indices.view(B, S)
        
        # Loss: ||sg(z) - e||^2 + beta * ||z - sg(e)||^2
        loss = F.mse_loss(z_q, z.detach()) + self.commitment_cost * F.mse_loss(z.detach(), z_q)
        
        # Straight-through gradient
        z_q_ste = z + (z_q - z).detach()
        
        return z_q_ste, loss, indices

    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(indices)


class MultiscaleResidualVQ(nn.Module):
    """
    Hierarchical residual vector quantizer over multiple sequence scales.
    Levels = [S_0, S_1, ..., S_{K-1}] where S_0 < S_1 < ... < S_{K-1} = L.
    """
    def __init__(self, levels: List[int], embed_dim: int, codebook_size: int = 256, commitment_cost: float = 0.25):
        super().__init__()
        self.levels = levels
        self.embed_dim = embed_dim
        self.K = len(levels)
        self.L = levels[-1]
        
        self.quantizers = nn.ModuleList([
            VectorQuantizer(codebook_size, embed_dim, commitment_cost)
            for _ in levels
        ])

    def downsample(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """[B, L, D] -> [B, target_len, D] using area interpolation"""
        if x.shape[1] == target_len:
            return x
        # Interpolate expects [B, D, L]
        return F.interpolate(x.transpose(1, 2), size=target_len, mode="area").transpose(1, 2)

    def upsample(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """[B, S, D] -> [B, target_len, D] using linear interpolation"""
        if x.shape[1] == target_len:
            return x
        return F.interpolate(x.transpose(1, 2), size=target_len, mode="linear", align_corners=False).transpose(1, 2)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            z_e: Continuous encoder output of shape [B, L, D]
        Returns:
            z_q_full: [B, L, D] accumulated multiscale quantized representation
            total_loss: VQ loss summed across levels
            all_indices: List of discrete indices [c_0, ..., c_{K-1}]
            z_intermediates: List of accumulated representations up to each level
        """
        B, L, D = z_e.shape
        z_hat = torch.zeros(B, L, D, device=z_e.device, dtype=z_e.dtype)
        
        total_loss = 0.0
        all_indices = []
        z_intermediates = []
        
        for k, (level_len, quantizer) in enumerate(zip(self.levels, self.quantizers)):
            # Residual at current level
            residual = z_e - z_hat
            # Downsample residual to level scale
            res_down = self.downsample(residual, level_len)
            # Quantize
            z_q_k, vq_loss_k, idx_k = quantizer(res_down)
            total_loss = total_loss + vq_loss_k
            all_indices.append(idx_k)
            
            # Upsample quantized residual to full length L and accumulate
            z_q_up = self.upsample(z_q_k, L)
            z_hat = z_hat + z_q_up
            z_intermediates.append(z_hat.clone())
            
        return z_hat, total_loss, all_indices, z_intermediates

    def codes_to_z(self, codes_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Reconstructs the continuous representations at each scale from discrete indices.
        """
        B = codes_list[0].shape[0]
        device = codes_list[0].device
        z_hat = torch.zeros(B, self.L, self.embed_dim, device=device)
        intermediates = []
        
        for k, (idx_k, quantizer) in enumerate(zip(codes_list, self.quantizers)):
            emb_k = quantizer.get_codebook_entry(idx_k)
            emb_up = self.upsample(emb_k, self.L)
            z_hat = z_hat + emb_up
            intermediates.append(z_hat.clone())
            
        return intermediates


from custom_ncm.rope import RotaryPositionalEmbedding

class CodeVQVAE(nn.Module):
    """
    Stage-1 Multiscale Concept VQ-VAE.
    Maps code tokens <-> hierarchical discrete concepts.
    Supports 1M Context Window via RoPE (Rotary Position Embeddings).
    """
    def __init__(
        self,
        vocab_size: int,
        context_length: int = 64,
        embed_dim: int = 128,
        levels: List[int] = [2, 8, 32, 64],
        codebook_size: int = 256,
        n_enc_layers: int = 2,
        n_dec_layers: int = 2,
        commitment_cost: float = 0.25,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embed_dim = embed_dim
        self.levels = levels
        self.use_rope = context_length > 8192
        
        # Token and positional embeddings
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        if self.use_rope:
            self.rope = RotaryPositionalEmbedding(embed_dim, max_seq_len=context_length)
            self.pos_emb = None
            self.dec_pos_emb = None
        else:
            self.pos_emb = nn.Embedding(context_length, embed_dim)
            self.dec_pos_emb = nn.Embedding(context_length, embed_dim)
            self.rope = None
        
        # Transformer Encoder
        self.enc_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_head=4, dropout=dropout)
            for _ in range(n_enc_layers)
        ])
        self.enc_ln = nn.LayerNorm(embed_dim)
        
        # Multiscale Residual Quantizer
        self.vq = MultiscaleResidualVQ(levels, embed_dim, codebook_size, commitment_cost)
        
        # Transformer Decoder
        self.dec_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_head=4, dropout=dropout)
            for _ in range(n_dec_layers)
        ])
        self.dec_ln = nn.LayerNorm(embed_dim)
        
        # Head predicting code token logits
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.token_emb.weight

    def get_pos_emb(self, x: torch.Tensor, L: int) -> torch.Tensor:
        if self.use_rope:
            return self.rope(x, L)
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        return self.pos_emb(pos)

    def get_dec_pos_emb(self, x: torch.Tensor, L: int) -> torch.Tensor:
        if self.use_rope:
            return self.rope(x, L)
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        return self.dec_pos_emb(pos)

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Returns the list of discrete concept indices across levels: [c_0, ..., c_{K-1}]"""
        B, L = x.shape
        h = self.token_emb(x) + self.get_pos_emb(x, L)
        for block in self.enc_blocks:
            h = block(h)
        h = self.enc_ln(h)
        
        _, _, indices_list, _ = self.vq(h)
        return indices_list

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: [B, L] token IDs
        Returns:
            logits: [B, L, Vocab] reconstruction logits
            vq_loss: scalar commitment loss
            indices: List of discrete concept code tensors
        """
        B, L = x.shape
        h = self.token_emb(x) + self.get_pos_emb(x, L)
        for block in self.enc_blocks:
            h = block(h)
        z_e = self.enc_ln(h)
        
        z_q, vq_loss, indices_list, _ = self.vq(z_e)
        
        # Straight-Through Estimator for decoder with positional embedding
        z_ste = z_e + (z_q - z_e).detach()
        h_dec = z_ste + self.get_dec_pos_emb(z_ste, L)
        for block in self.dec_blocks:
            h_dec = block(h_dec)
        logits = self.lm_head(self.dec_ln(h_dec))
        
        return logits, vq_loss, indices_list

    def decode_from_level(self, intermediates: List[torch.Tensor], level_idx: int) -> torch.Tensor:
        """Reconstructs token logits from the accumulated representation at a specific level."""
        z_k = intermediates[level_idx]
        L = z_k.shape[1]
        h_dec = z_k + self.get_dec_pos_emb(z_k, L)
        for block in self.dec_blocks:
            h_dec = block(h_dec)
        return self.lm_head(self.dec_ln(h_dec))

    def decode_multiscales(self, codes_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Decodes all levels progressively to visualize coarse-to-fine refinement.
        Returns a list of token predictions at each scale k: 0, 1, ..., K-1.
        """
        intermediates = self.vq.codes_to_z(codes_list)
        outputs = []
        for k in range(len(intermediates)):
            logits = self.decode_from_level(intermediates, k)
            pred_tokens = torch.argmax(logits, dim=-1)
            outputs.append(pred_tokens)
        return outputs
