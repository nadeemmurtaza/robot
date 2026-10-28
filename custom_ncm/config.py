"""
Configuration dataclasses for Custom NCM (Next-Concept Modeling) for Code.
Supports configurable model parameter scaling:
- Small  (~2.9M params): Fast, lightweight experimentation
- Medium (~19.3M params): High-capacity representation for complex functions
- Large  (~76.5M params): Deep multi-scale architecture
"""
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class NCMConfig:
    # Sequence and scale lengths
    context_length: int = 64          # Maximum target code tokens (L)
    prefix_length: int = 32           # Maximum prompt/prefix tokens (P)
    
    # Hierarchy of concept levels:
    # E.g. [2, 8, 32, 64] means:
    # Level 0: 2 tokens (global coarse concept)
    # Level 1: 8 tokens (structural syntax outline)
    # Level 2: 32 tokens (statement and block details)
    # Level 3: 64 tokens (exact fine tokens)
    levels: List[int] = field(default_factory=lambda: [2, 8, 32, 64])
    
    # Vocabulary & Embedding
    vocab_size: int = 256             # Code tokenizer vocabulary size
    embed_dim: int = 128              # Dimension of concept vector space (d)
    
    # VQ-VAE (Concept Tokenizer) settings
    vq_codebook_size: int = 256       # Discrete codes per level
    commitment_cost: float = 0.25     # Beta in VQ loss
    vq_decay: float = 0.99            # EMA decay for codebook updates
    vq_enc_layers: int = 2            # Transformer encoder layers in VQ-VAE
    vq_dec_layers: int = 2            # Transformer decoder layers in VQ-VAE
    
    # Next-Concept Generator (Transformer) settings
    gen_n_layer: int = 4              # Layers in generator transformer
    gen_n_head: int = 4               # Attention heads
    gen_dropout: float = 0.1
    gen_use_bias: bool = True
    
    # Training settings
    batch_size: int = 16
    lr_vqvae: float = 1e-3
    lr_gen: float = 5e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    
    # Generation settings
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9

    @classmethod
    def small(cls) -> "NCMConfig":
        """Small tier (~2.9M params) - Default lightweight baseline."""
        return cls(
            context_length=64,
            prefix_length=32,
            levels=[2, 8, 32, 64],
            embed_dim=128,
            vq_codebook_size=256,
            vq_enc_layers=2,
            vq_dec_layers=2,
            gen_n_layer=4,
            gen_n_head=4
        )

    @classmethod
    def medium(cls) -> "NCMConfig":
        """Medium tier (~19.3M params) - Scaled capacity for complex algorithms and statements."""
        return cls(
            context_length=128,
            prefix_length=64,
            levels=[4, 16, 64, 128],
            embed_dim=256,
            vq_codebook_size=512,
            vq_enc_layers=4,
            vq_dec_layers=4,
            gen_n_layer=6,
            gen_n_head=8
        )

    @classmethod
    def large(cls) -> "NCMConfig":
        """Large tier (~76.5M params) - Maximum capacity for multi-statement functions."""
        return cls(
            context_length=256,
            prefix_length=64,
            levels=[4, 16, 64, 256],
            embed_dim=512,
            vq_codebook_size=1024,
            vq_enc_layers=6,
            vq_dec_layers=6,
            gen_n_layer=8,
            gen_n_head=8
        )
