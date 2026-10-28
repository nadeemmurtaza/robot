"""
Complete End-to-End Pipeline for Custom Next-Concept Modeling (NCM) for Code.
Executes:
1. Tokenizer & Dataset Verification
2. Stage-1: Multiscale Concept VQ-VAE Training
3. Stage-2: Next-Concept Transformer Generator Training
4. Inference & Step-by-Step Coarse-to-Fine Generation Demo
"""
import os
import sys
import torch

from custom_ncm.config import NCMConfig
from custom_ncm.code_tokenizer import ByteCodeTokenizer
from custom_ncm.dataset import get_dataloaders
from custom_ncm.train_vqvae import train_vqvae
from custom_ncm.train_ncm import train_ncm
from custom_ncm.generate import generate_code_coarse_to_fine

def main():
    print("=" * 70)
    print("  BUILDING & TRAINING A CUSTOM NEXT-CONCEPT MODEL (NCM) FOR CODING")
    print("  Inspired by Stanford Hazy Research (Parallel Coarse-to-Fine Generation)")
    print("=" * 70)
    
    # 1. Configuration
    config = NCMConfig(
        context_length=64,           # Code completion length
        prefix_length=32,            # Prompt length
        levels=[2, 8, 32, 64],       # 4 hierarchy scales: Coarse -> Structural -> Statement -> Token
        embed_dim=128,               # Latent concept dimension
        vq_codebook_size=256,        # Discrete codebook size per level
        vq_enc_layers=2,
        vq_dec_layers=2,
        gen_n_layer=3,
        gen_n_head=4,
        batch_size=16,
        lr_vqvae=1.5e-3,
        lr_gen=1e-3
    )
    
    print("\n--- 1. CONFIGURATION ---")
    print(f"  Target Context Length (L): {config.context_length}")
    print(f"  Prompt Prefix Length (P): {config.prefix_length}")
    print(f"  Concept Hierarchy Levels: {config.levels}")
    print(f"  Latent Embed Dim: {config.embed_dim}")
    print(f"  Codebook Size: {config.vq_codebook_size}")
    
    # 2. Stage 1: Train VQ-VAE (Concept Tokenizer)
    print("\n--- 2. STAGE 1: TRAINING MULTISCALE CONCEPT VQ-VAE ---")
    vqvae_ckpt = "checkpoints/vqvae.pt"
    vqvae = train_vqvae(config, epochs=12, checkpoint_path=vqvae_ckpt)
    
    # 3. Stage 2: Train Generator (Next-Concept Transformer)
    print("\n--- 3. STAGE 2: TRAINING NEXT-CONCEPT GENERATOR ---")
    gen_ckpt = "checkpoints/generator.pt"
    generator = train_ncm(config, vqvae=vqvae, gen_checkpoint=gen_ckpt, epochs=15)
    
    # 4. Coarse-to-Fine Code Generation Showcase
    print("\n--- 4. INFERENCE: COARSE-TO-FINE CODE GENERATION DEMO ---")
    tokenizer = ByteCodeTokenizer()
    
    test_prompts = [
        "# Compute the sum of all elements\ndef sum_list(nums):\n",
        "# Return the maximum value in a list\ndef get_max(nums):\n",
        "# Check if a string is a palindrome\ndef is_palindrome(s):\n",
        "# Filter even numbers from list\ndef filter_evens(nums):\n"
    ]
    
    for prompt in test_prompts:
        generate_code_coarse_to_fine(
            generator=generator,
            tokenizer=tokenizer,
            prompt=prompt,
            prefix_len=config.prefix_length,
            temperature=0.7,
            top_k=20,
            verbose=True
        )

if __name__ == "__main__":
    main()
