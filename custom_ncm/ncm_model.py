"""
Next-Concept Transformer Generator.
Predicts discrete concept codes level-by-level in parallel.
Each level is generated in a single forward evaluation.
"""
from typing import List, Tuple, Dict, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_ncm.vqvae import CodeVQVAE
from custom_ncm.rope import RotaryPositionalEmbedding

class NCMTransformerBlock(nn.Module):
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
        # attn_mask has shape [SeqLen, SeqLen] with -inf for forbidden positions
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class NextConceptGenerator(nn.Module):
    """
    Coarse-to-fine Transformer Generator.
    Conditions on prompt/prefix code tokens and generates the hierarchy of concepts.
    """
    def __init__(
        self,
        vqvae: CodeVQVAE,
        prefix_len: int = 32,
        d_model: int = 128,
        n_head: int = 4,
        n_layer: int = 4,
        dropout: float = 0.1,
        codebook_size: int = 256
    ):
        super().__init__()
        self.vqvae = vqvae
        self.prefix_len = prefix_len
        self.d_model = d_model
        self.levels = vqvae.levels
        self.K = len(self.levels)
        self.codebook_size = codebook_size
        
        # Prefix embedding (for prompt tokens)
        self.prefix_token_emb = nn.Embedding(vqvae.vocab_size, d_model)
        if prefix_len > 8192:
            self.prefix_rope = RotaryPositionalEmbedding(d_model, max_seq_len=prefix_len)
            self.prefix_pos_emb = None
        else:
            self.prefix_pos_emb = nn.Embedding(prefix_len, d_model)
            self.prefix_rope = None
        
        # Level 0 learned query vectors
        self.l0_queries = nn.Parameter(torch.randn(1, self.levels[0], d_model) * 0.02)
        
        # Level embeddings to distinguish hierarchy levels
        self.level_emb = nn.Embedding(self.K, d_model)
        # Positional embedding for positions within any level
        max_l = max(self.levels)
        if max_l > 8192:
            self.level_rope = RotaryPositionalEmbedding(d_model, max_seq_len=max_l)
            self.level_pos_emb = None
        else:
            self.level_pos_emb = nn.Embedding(max_l, d_model)
            self.level_rope = None
        
        # Projection from VQ-VAE embed_dim to generator d_model if different
        if vqvae.embed_dim != d_model:
            self.input_proj = nn.Linear(vqvae.embed_dim, d_model)
        else:
            self.input_proj = nn.Identity()
            
        # Transformer backbone
        self.blocks = nn.ModuleList([
            NCMTransformerBlock(d_model, n_head=n_head, dropout=dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        
        # Per-level output prediction heads (each level predicts codebook indices)
        self.heads = nn.ModuleList([
            nn.Linear(d_model, codebook_size)
            for _ in self.levels
        ])

    def get_prefix_pos(self, x: torch.Tensor, P: int) -> torch.Tensor:
        if self.prefix_rope is not None:
            return self.prefix_rope(x, P)
        pos = torch.arange(P, device=x.device).unsqueeze(0)
        return self.prefix_pos_emb(pos)

    def get_level_pos(self, x: torch.Tensor, L: int) -> torch.Tensor:
        if self.level_rope is not None:
            return self.level_rope(x, L)
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        return self.level_pos_emb(pos)

    def build_block_causal_mask(self, P: int, levels: List[int], device: torch.device) -> torch.Tensor:
        """
        Constructs the hierarchical attention mask:
        - Prefix can attend to Prefix
        - Level l can attend to Prefix and previous levels (< l)
        - Within Level l, attention is FULLY BIDIRECTIONAL (all positions in parallel)
        - Level l CANNOT attend to future levels (> l)
        """
        total_len = P + sum(levels)
        mask = torch.full((total_len, total_len), float('-inf'), device=device)
        
        # Prefix self-attention
        mask[:P, :P] = 0.0
        
        # Block layout
        start_q = P
        for l_idx, l_len in enumerate(levels):
            end_q = start_q + l_len
            
            # Can attend to Prefix
            mask[start_q:end_q, :P] = 0.0
            
            # Can attend to all previous levels and current level
            mask[start_q:end_q, P:end_q] = 0.0
            
            start_q = end_q
            
        return mask

    def forward(
        self,
        prefix_ids: torch.Tensor,
        target_ids: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        Training forward pass.
        Given prefix_ids and target_ids, extracts target concept codes from frozen VQ-VAE,
        feeds the hierarchical representation through the transformer with block-causal mask,
        and computes cross-entropy loss against the true concept codes at each level.
        """
        B = prefix_ids.shape[0]
        device = prefix_ids.device
        
        # 1. Extract ground-truth concept codes and intermediate representations from VQ-VAE
        with torch.no_grad():
            B_tgt, L_tgt = target_ids.shape
            h_tgt = self.vqvae.token_emb(target_ids) + self.vqvae.get_pos_emb(target_ids, L_tgt)
            for block in self.vqvae.enc_blocks:
                h_tgt = block(h_tgt)
            z_e = self.vqvae.enc_ln(h_tgt)
            
            _, _, target_codes, z_intermediates = self.vqvae.vq(z_e)
            
        # 2. Encode prefix prompt
        P = prefix_ids.shape[1]
        h_prefix = self.prefix_token_emb(prefix_ids) + self.get_prefix_pos(prefix_ids, P)
        
        # 3. Assemble inputs for each level
        # Level 0 input: learned queries + level 0 embedding + positional embedding
        l0_in = self.l0_queries.expand(B, -1, -1) + self.level_emb(torch.tensor(0, device=device)) + self.get_level_pos(self.l0_queries, self.levels[0])
        
        level_inputs = [l0_in]
        for l in range(1, self.K):
            # Input to level l is accumulated representation from previous level (l-1)
            # downsampled/interpolated to length levels[l]
            z_prev = z_intermediates[l - 1]
            z_down = self.vqvae.vq.downsample(z_prev, self.levels[l])
            z_proj = self.input_proj(z_down) + self.level_emb(torch.tensor(l, device=device)) + self.get_level_pos(z_down, self.levels[l])
            level_inputs.append(z_proj)
            
        # Concatenate full sequence: [Prefix, Level 0, Level 1, ..., Level K-1]
        full_seq = torch.cat([h_prefix] + level_inputs, dim=1)
        
        # Build block-causal attention mask
        attn_mask = self.build_block_causal_mask(self.prefix_len, self.levels, device)
        
        # Transformer forward
        h = full_seq
        for block in self.blocks:
            h = block(h, attn_mask=attn_mask)
        h = self.ln_f(h)
        
        # Compute loss and collect per-level logits
        all_logits = []
        total_loss = 0.0
        
        offset = self.prefix_len
        for l, (level_len, head) in enumerate(zip(self.levels, self.heads)):
            level_h = h[:, offset:offset + level_len, :]
            logits = head(level_h) # [B, level_len, CodebookSize]
            all_logits.append(logits)
            
            # Cross entropy against ground-truth codes for this level
            loss_l = F.cross_entropy(logits.reshape(-1, self.codebook_size), target_codes[l].reshape(-1))
            total_loss = total_loss + loss_l
            
            offset += level_len
            
        return all_logits, target_codes, total_loss

    @torch.no_grad()
    def generate(
        self,
        prefix_ids: torch.Tensor,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Hierarchical Parallel Generation (Coarse to Fine).
        Takes EXACTLY K forward evaluations (e.g. 4 steps) to generate a full code snippet!
        Returns:
            predicted_codes: list of concept codes [c_0, ..., c_{K-1}]
            decoded_progression: reconstructed token sequences at each level 0..K-1
        """
        self.eval()
        B = prefix_ids.shape[0]
        device = prefix_ids.device
        
        # 1. Encode prefix
        P = prefix_ids.shape[1]
        h_prefix = self.prefix_token_emb(prefix_ids) + self.get_prefix_pos(prefix_ids, P)
        
        predicted_codes = []
        z_hat = torch.zeros(B, self.vqvae.context_length, self.vqvae.embed_dim, device=device)
        
        for l in range(self.K):
            level_len = self.levels[l]
            
            # Prepare input for level l with level embedding and intra-level position encoding
            if l == 0:
                l_in = self.l0_queries.expand(B, -1, -1) + self.level_emb(torch.tensor(0, device=device)) + self.get_level_pos(self.l0_queries, self.levels[0])
            else:
                z_down = self.vqvae.vq.downsample(z_hat, level_len)
                l_in = self.input_proj(z_down) + self.level_emb(torch.tensor(l, device=device)) + self.get_level_pos(z_down, level_len)
                
            # Concatenate prefix + current level input
            # Attention allows bidirectional attention within current level and cross-attention to prefix
            seq = torch.cat([h_prefix, l_in], dim=1)
            
            # Mask: prefix attends to prefix; current level attends to prefix + current level
            seq_len = P + level_len
            gen_mask = torch.zeros((seq_len, seq_len), device=device)
            gen_mask[:P, P:] = float('-inf')
            
            h = seq
            for block in self.blocks:
                h = block(h, attn_mask=gen_mask)
            h = self.ln_f(h)
            
            # Extract current level hidden states and project with level head
            level_h = h[:, P:, :]
            logits = self.heads[l](level_h) # [B, level_len, CodebookSize]
            
            # Sample discrete codes for this level in parallel!
            codes_l = self._sample(logits, temperature, top_k, top_p)
            predicted_codes.append(codes_l)
            
            # Lookup codebook entries and accumulate into z_hat
            emb_l = self.vqvae.vq.quantizers[l].get_codebook_entry(codes_l)
            emb_up = self.vqvae.vq.upsample(emb_l, self.vqvae.context_length)
            z_hat = z_hat + emb_up
            
        # Decode the multiscale progression through VQ-VAE decoder
        progression = self.vqvae.decode_multiscales(predicted_codes)
        return predicted_codes, progression

    def _sample(self, logits: torch.Tensor, temperature: float = 0.8, top_k: int = 40, top_p: float = 0.9) -> torch.Tensor:
        if temperature <= 0.0:
            return torch.argmax(logits, dim=-1)
            
        logits = logits / temperature
        B, S, V = logits.shape
        flat_logits = logits.reshape(-1, V)
        
        # Top-k filtering
        if top_k > 0:
            indices_to_remove = flat_logits < torch.topk(flat_logits, top_k)[0][..., -1, None]
            flat_logits[indices_to_remove] = -float('Inf')
            
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(flat_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            flat_logits[indices_to_remove] = -float('Inf')
            
        probs = F.softmax(flat_logits, dim=-1)
        samples = torch.multinomial(probs, num_samples=1).view(B, S)
        return samples
