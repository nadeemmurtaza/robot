"""
Training Pipeline for the Hybrid Consistency Next-Concept Model (HC-NCM).
Jointly trains:
1. LCM Latent Consistency Planner (Diffusion Gemma bidirectional attention)
2. NCM Hierarchical Concept Generator
3. Causal Gemma-style Token Decoder Head
"""
import os
import sys
import json
import time
from typing import Optional, List, Tuple
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from custom_ncm.config import NCMConfig
from custom_ncm.code_tokenizer import ByteCodeTokenizer
from custom_ncm.dataset import CodeDataset
from custom_ncm.vqvae import CodeVQVAE
from custom_ncm.ncm_model import NextConceptGenerator
from custom_ncm.hybrid_model import HybridConsistencyNCM
from custom_ncm.official_python_harvester import OfficialPythonHarvester
from custom_ncm.attention import convert_legacy_mha_state_dict

def train_hybrid_model(
    config: Optional[NCMConfig] = None,
    epochs: int = 5,
    batch_size: int = 16,
    checkpoint_dir: str = "checkpoints/hybrid",
    device: Optional[torch.device] = None
) -> HybridConsistencyNCM:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config is None:
        config = NCMConfig.small()
        
    print("=" * 70)
    print("  TRAINING HYBRID CONSISTENCY NEXT-CONCEPT MODEL (HC-NCM)")
    print("  LCM (1-Step Consistency) + Diffusion Gemma (Bidirectional) + NCM + Causal AR")
    print("=" * 70)
    print(f"[*] Compute Device: {device}")
    
    tokenizer = ByteCodeTokenizer()
    
    # 1. Load VQ-VAE
    vqvae_ckpt = "checkpoints/vqvae_pos.pt" if os.path.exists("checkpoints/vqvae_pos.pt") else "checkpoints/vqvae.pt"
    if not os.path.exists(vqvae_ckpt):
        from custom_ncm.train_vqvae import train_vqvae
        vqvae = train_vqvae(config, epochs=6, checkpoint_path=vqvae_ckpt)
    else:
        v_data = torch.load(vqvae_ckpt, map_location=device, weights_only=False)
        vqvae = CodeVQVAE(
            vocab_size=tokenizer.vocab_size,
            context_length=config.context_length,
            embed_dim=config.embed_dim,
            levels=config.levels,
            codebook_size=config.vq_codebook_size,
            n_enc_layers=config.vq_enc_layers,
            n_dec_layers=config.vq_dec_layers
        ).to(device)
        vqvae.load_state_dict(convert_legacy_mha_state_dict(v_data["model"]))
        print(f"[+] Loaded VQ-VAE Concept Tokenizer from: {vqvae_ckpt}")
        
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False
        
    # 2. Load Base Generator
    gen_ckpt = "checkpoints/generator_official_python.pt"
    if not os.path.exists(gen_ckpt):
        gen_ckpt = "checkpoints/generator_base.pt"
    if not os.path.exists(gen_ckpt):
        gen_ckpt = "checkpoints/generator.pt"
        
    ncm_gen = NextConceptGenerator(
        vqvae=vqvae,
        prefix_len=config.prefix_length,
        d_model=config.embed_dim,
        n_head=config.gen_n_head,
        n_layer=config.gen_n_layer,
        codebook_size=config.vq_codebook_size
    ).to(device)
    
    if os.path.exists(gen_ckpt):
        g_data = torch.load(gen_ckpt, map_location=device, weights_only=False)
        ncm_gen.load_state_dict(convert_legacy_mha_state_dict(g_data["model"]), strict=False)
        print(f"[+] Loaded NCM Generator baseline weights from: {gen_ckpt}")
        
    # 3. Initialize Hybrid Model
    hybrid_model = HybridConsistencyNCM(
        config=config,
        vqvae=vqvae,
        ncm_generator=ncm_gen
    ).to(device)
    
    # 4. Load Training Corpus
    corpus_file = "data/official_python_corpus.json"
    if not os.path.exists(corpus_file):
        harvester = OfficialPythonHarvester()
        corpus = harvester.harvest_all(corpus_file)
    else:
        with open(corpus_file, "r", encoding="utf-8") as f:
            corpus = json.load(f)
            
    pairs = [(item["prompt"], item["body"]) for item in corpus]
    dataset = CodeDataset(pairs, tokenizer, prefix_len=config.prefix_length, context_len=config.context_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 5. Optimizer
    optimizer = AdamW(
        [
            {"params": hybrid_model.lcm_planner.parameters(), "lr": 1e-3},
            {"params": hybrid_model.ncm_generator.parameters(), "lr": 3e-4},
            {"params": hybrid_model.causal_blocks.parameters(), "lr": 8e-4},
            {"params": hybrid_model.token_head.parameters(), "lr": 8e-4}
        ],
        weight_decay=0.01
    )
    
    # 6. Training Loop
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, "hybrid_model.pt")
    
    print(f"\n[*] Commencing {epochs} epochs of Hybrid Consistency Training...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        hybrid_model.train()
        total_loss = 0.0
        total_lcm = 0.0
        total_token = 0.0
        
        for batch in loader:
            prefix = batch["prefix"].to(device)
            target = batch["target"].to(device)
            
            optimizer.zero_grad()
            losses = hybrid_model(prefix, target)
            loss = losses["total_loss"]
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(hybrid_model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * prefix.size(0)
            total_lcm += losses["lcm_loss"].item() * prefix.size(0)
            total_token += losses["token_loss"].item() * prefix.size(0)
            
        n = len(dataset)
        avg_loss = total_loss / n
        avg_lcm = total_lcm / n
        avg_token = total_token / n
        print(f"Epoch {epoch:02d}/{epochs} | Total Loss: {avg_loss:.4f} | LCM Loss: {avg_lcm:.4f} | Causal Token Loss: {avg_token:.4f}", flush=True)
        
    torch.save({
        "model": hybrid_model.state_dict(),
        "config": config,
        "lcm_planner": hybrid_model.lcm_planner.state_dict(),
        "causal_blocks": hybrid_model.causal_blocks.state_dict()
    }, save_path)
    
    print(f"[+] Hybrid Model trained in {time.time() - start_time:.1f}s. Saved to: {save_path}\n")
    return hybrid_model

if __name__ == "__main__":
    train_hybrid_model(epochs=3)
