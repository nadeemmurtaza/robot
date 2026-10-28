"""
Trains the Next-Concept Model directly on the Official Python Standard Library & Docs.
Ingests real-world functions (bisect, heapq, statistics, random, functools, string)
and teaches the model official Python conventions, algorithms, and idioms.
"""
import os
import sys
import json
import time
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

torch.set_num_threads(8)

from custom_ncm.config import NCMConfig
from custom_ncm.code_tokenizer import ByteCodeTokenizer
from custom_ncm.dataset import CodeDataset
from custom_ncm.vqvae import CodeVQVAE
from custom_ncm.ncm_model import NextConceptGenerator
from custom_ncm.generate import generate_code_coarse_to_fine
from custom_ncm.official_python_harvester import OfficialPythonHarvester

def main(epochs: int = 8, checkpoint_path: str = "checkpoints/generator_official_python.pt"):
    print("=" * 70, flush=True)
    print("  TRAINING NCM MODEL ON OFFICIAL PYTHON STANDARD LIBRARY & DOCS", flush=True)
    print("=" * 70, flush=True)
    
    corpus_file = "data/official_python_corpus.json"
    if not os.path.exists(corpus_file):
        harvester = OfficialPythonHarvester()
        corpus = harvester.harvest_all(corpus_file)
    else:
        with open(corpus_file, "r", encoding="utf-8") as f:
            corpus = json.load(f)
            
    print(f"[*] Loaded {len(corpus)} official Python functions into training corpus.", flush=True)
    
    # Extract pairs
    pairs = [(item["prompt"], item["body"]) for item in corpus]
    
    tokenizer = ByteCodeTokenizer()
    config = NCMConfig(
        context_length=64,
        prefix_length=32,
        levels=[2, 8, 32, 64],
        embed_dim=128,
        vq_codebook_size=256
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}", flush=True)
    
    # Load VQ-VAE
    vqvae_ckpt = "checkpoints/vqvae_pos.pt"
    if not os.path.exists(vqvae_ckpt):
        print(f"[!] {vqvae_ckpt} not found. Running VQ-VAE training first...")
        from custom_ncm.train_vqvae import train_vqvae
        vqvae = train_vqvae(config, epochs=8, checkpoint_path=vqvae_ckpt)
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
        vqvae.load_state_dict(v_data["model"])
        
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False
        
    # Load Base Generator
    gen_ckpt = "checkpoints/generator_base.pt"
    generator = NextConceptGenerator(
        vqvae=vqvae,
        prefix_len=config.prefix_length,
        d_model=config.embed_dim,
        n_head=config.gen_n_head,
        n_layer=config.gen_n_layer,
        codebook_size=config.vq_codebook_size
    ).to(device)
    
    if os.path.exists(gen_ckpt):
        g_data = torch.load(gen_ckpt, map_location=device, weights_only=False)
        generator.load_state_dict(g_data["model"], strict=False)
        print(f"[+] Loaded baseline weights from {gen_ckpt}", flush=True)
        
    # Create DataLoader
    dataset = CodeDataset(pairs, tokenizer, prefix_len=config.prefix_length, context_len=config.context_length)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    optimizer = AdamW(generator.parameters(), lr=config.lr_gen * 0.7, weight_decay=config.weight_decay)
    
    print(f"\n[*] Commencing {epochs} epochs of Official Python Knowledge Ingestion...", flush=True)
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        generator.train()
        total_loss = 0.0
        
        for batch in loader:
            prefix = batch["prefix"].to(device)
            target = batch["target"].to(device)
            
            optimizer.zero_grad()
            _, _, loss = generator(prefix, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * prefix.size(0)
            
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch:02d}/{epochs} | Official Python Training Loss: {avg_loss:.4f}", flush=True)
        
    torch.save({"model": generator.state_dict(), "config": config}, checkpoint_path)
    print(f"[+] Finished in {time.time() - start_time:.1f}s. Saved to {checkpoint_path}\n", flush=True)
    
    # Sample Generation from Official Python Tasks
    print("=" * 70, flush=True)
    print("  TESTING MODEL ON OFFICIAL PYTHON STANDARD LIBRARY PROMPTS", flush=True)
    print("=" * 70, flush=True)
    
    test_prompts = [
        "# Return index where to insert item x in list a, keeping sorted\ndef bisect_right(a, x):\n",
        "# Pop smallest item off heap and return it\ndef heappop(heap):\n",
        "# Return arithmetic mean of data\ndef mean(data):\n"
    ]
    
    for prompt in test_prompts:
        generate_code_coarse_to_fine(
            generator=generator,
            tokenizer=tokenizer,
            prompt=prompt,
            prefix_len=config.prefix_length,
            temperature=0.3,
            top_k=20,
            verbose=True
        )

if __name__ == "__main__":
    main(epochs=6)
