"""
Stage 2 Training: Next-Concept Transformer Generator.
Trains the generator to predict discrete concept codes level-by-level in parallel given a code prompt.
"""
import os
import time
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from custom_ncm.config import NCMConfig
from custom_ncm.code_tokenizer import ByteCodeTokenizer
from custom_ncm.dataset import get_dataloaders
from custom_ncm.vqvae import CodeVQVAE
from custom_ncm.ncm_model import NextConceptGenerator

def train_ncm(
    config: NCMConfig,
    vqvae: CodeVQVAE = None,
    vqvae_checkpoint: str = "checkpoints/vqvae.pt",
    gen_checkpoint: str = "checkpoints/generator.pt",
    epochs: int = 20
):
    os.makedirs(os.path.dirname(gen_checkpoint), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training NCM Generator on device: {device}")
    
    tokenizer = ByteCodeTokenizer()
    train_loader, val_loader = get_dataloaders(
        tokenizer,
        prefix_len=config.prefix_length,
        context_len=config.context_length,
        batch_size=config.batch_size
    )
    
    # 1. Load or prepare frozen VQ-VAE
    if vqvae is None:
        if not os.path.exists(vqvae_checkpoint):
            raise FileNotFoundError(f"VQ-VAE checkpoint not found at {vqvae_checkpoint}. Train VQ-VAE first!")
        ckpt = torch.load(vqvae_checkpoint, map_location=device, weights_only=False)
        vqvae = CodeVQVAE(
            vocab_size=tokenizer.vocab_size,
            context_length=config.context_length,
            embed_dim=config.embed_dim,
            levels=config.levels,
            codebook_size=config.vq_codebook_size,
            n_enc_layers=config.vq_enc_layers,
            n_dec_layers=config.vq_dec_layers
        ).to(device)
        vqvae.load_state_dict(ckpt["model"])
        
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False
        
    # 2. Initialize NextConceptGenerator
    generator = NextConceptGenerator(
        vqvae=vqvae,
        prefix_len=config.prefix_length,
        d_model=config.embed_dim,
        n_head=config.gen_n_head,
        n_layer=config.gen_n_layer,
        dropout=config.gen_dropout,
        codebook_size=config.vq_codebook_size
    ).to(device)
    
    optimizer = AdamW(generator.parameters(), lr=config.lr_gen, weight_decay=config.weight_decay)
    
    print(f"[*] Generator Parameters: {sum(p.numel() for p in generator.parameters()) / 1e3:.1f}k")
    print(f"[*] Parallel generation steps required: {len(config.levels)} (vs {config.context_length} autoregressive steps)")
    
    start_time = time.time()
    best_val_loss = float("inf")
    
    for epoch in range(1, epochs + 1):
        generator.train()
        total_loss = 0.0
        
        for batch in train_loader:
            prefix = batch["prefix"].to(device)
            target = batch["target"].to(device)
            
            optimizer.zero_grad()
            _, _, loss = generator(prefix, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), config.grad_clip)
            optimizer.step()
            
            total_loss += loss.item() * prefix.size(0)
            
        avg_loss = total_loss / len(train_loader.dataset)
        
        # Validation
        generator.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                prefix = batch["prefix"].to(device)
                target = batch["target"].to(device)
                _, _, v_loss = generator(prefix, target)
                val_loss += v_loss.item() * prefix.size(0)
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch:02d}/{epochs} | Train Loss (NCM cross-entropy): {avg_loss:.4f} | Val Loss: {val_loss:.4f}", flush=True)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model": generator.state_dict(), "config": config}, gen_checkpoint)
            
    print(f"[+] Generator Training finished in {time.time() - start_time:.1f}s. Saved best model to {gen_checkpoint}", flush=True)
    return generator

if __name__ == "__main__":
    train_ncm(NCMConfig(), epochs=15)
