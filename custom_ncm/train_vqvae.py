"""
Stage 1 Training: Multiscale Concept VQ-VAE for Code.
Learns to represent code sequences as hierarchical discrete concept codes.
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

def get_device() -> torch.device:
    """
    cuda > TPU (via torch_xla, if installed) > cpu.
    torch_xla.core.xla_model.xla_device() was deprecated starting with
    torch_xla 2.8 in favor of the top-level torch_xla.device() - try that
    first, fall back to the legacy API for older installs, and catch more
    than just ImportError since a real TPU runtime can fail here with other
    exception types (e.g. a PJRT init error) that shouldn't silently and
    invisibly downgrade an intended TPU run to CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    errors = []
    try:
        import torch_xla
        return torch_xla.device()
    except Exception as e:
        errors.append(f"torch_xla.device(): {e}")
    try:
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    except Exception as e:
        errors.append(f"torch_xla.core.xla_model.xla_device(): {e}")
    if errors:
        print(
            "  [!] torch_xla import/device calls both failed, falling back to CPU:\n    "
            + "\n    ".join(errors)
        )
    return torch.device("cpu")

def _warn_if_hardware_insufficient(config: NCMConfig, device: torch.device):
    is_tpu = "xla" in str(device)
    if config.context_length > 100_000 and device.type != "cuda" and not is_tpu:
        print(
            f"  [!] WARNING: context_length={config.context_length:,} on device '{device}' "
            "(no GPU/TPU detected). This tier was sized for a TPU v5e-1 / A100-class "
            "accelerator with tens of GB of RAM - it will likely exhaust memory or run "
            "for an unreasonable amount of time on CPU."
        )

def train_vqvae(config: NCMConfig, epochs: int = 15, checkpoint_path: str = "checkpoints/vqvae.pt"):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    device = get_device()
    print(f"[*] Training VQ-VAE on device: {device}")
    _warn_if_hardware_insufficient(config, device)
    
    tokenizer = ByteCodeTokenizer()
    train_loader, val_loader = get_dataloaders(
        tokenizer,
        prefix_len=config.prefix_length,
        context_len=config.context_length,
        batch_size=config.batch_size
    )
    
    model = CodeVQVAE(
        vocab_size=tokenizer.vocab_size,
        context_length=config.context_length,
        embed_dim=config.embed_dim,
        levels=config.levels,
        codebook_size=config.vq_codebook_size,
        n_enc_layers=config.vq_enc_layers,
        n_dec_layers=config.vq_dec_layers,
        commitment_cost=config.commitment_cost,
        dropout=0.1,
        attn_window=config.attn_window,
        attn_num_global=config.attn_num_global
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=config.lr_vqvae, weight_decay=config.weight_decay)
    
    print(f"[*] Total VQ-VAE Parameters: {sum(p.numel() for p in model.parameters()) / 1e3:.1f}k")
    print(f"[*] Hierarchy Levels: {config.levels} (Context length: {config.context_length})")
    
    start_time = time.time()
    best_loss = float("inf")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_recon_loss = 0.0
        total_vq_loss = 0.0
        correct_tokens = 0
        total_tokens = 0
        
        for batch in train_loader:
            targets = batch["target"].to(device) # [B, L]
            optimizer.zero_grad()
            
            logits, vq_loss, _ = model(targets)

            recon_loss = F.cross_entropy(
                logits.reshape(-1, tokenizer.vocab_size), targets.reshape(-1),
                ignore_index=tokenizer.pad_token_id
            )
            loss = recon_loss + vq_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            total_recon_loss += recon_loss.item() * targets.size(0)
            total_vq_loss += vq_loss.item() * targets.size(0)

            preds = torch.argmax(logits, dim=-1)
            non_pad = targets != tokenizer.pad_token_id
            correct_tokens += ((preds == targets) & non_pad).sum().item()
            total_tokens += non_pad.sum().item()
            
        n_samples = len(train_loader.dataset)
        avg_recon = total_recon_loss / n_samples
        avg_vq = total_vq_loss / n_samples
        accuracy = (correct_tokens / total_tokens) * 100.0
        
        # Validation
        model.eval()
        val_recon = 0.0
        with torch.no_grad():
            for batch in val_loader:
                targets = batch["target"].to(device)
                logits, _, _ = model(targets)
                v_loss = F.cross_entropy(
                    logits.reshape(-1, tokenizer.vocab_size), targets.reshape(-1),
                    ignore_index=tokenizer.pad_token_id
                )
                val_recon += v_loss.item() * targets.size(0)
        val_recon /= len(val_loader.dataset)
        
        print(f"Epoch {epoch:02d}/{epochs} | Train Recon: {avg_recon:.4f} | VQ Loss: {avg_vq:.4f} | Acc: {accuracy:.1f}% | Val Recon: {val_recon:.4f}", flush=True)
        
        if val_recon < best_loss:
            best_loss = val_recon
            torch.save({"model": model.state_dict(), "config": config}, checkpoint_path)
            
    print(f"[+] VQ-VAE Training finished in {time.time() - start_time:.1f}s. Saved best model to {checkpoint_path}", flush=True)
    return model

if __name__ == "__main__":
    import argparse
    from custom_ncm.config import add_tier_cli_arg
    parser = argparse.ArgumentParser(description="Train the Stage-1 VQ-VAE concept tokenizer")
    add_tier_cli_arg(parser)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vqvae.pt")
    args = parser.parse_args()
    train_vqvae(NCMConfig.from_tier(args.tier), epochs=args.epochs, checkpoint_path=args.checkpoint)
