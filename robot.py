"""
Code Robot: Next-Concept Modeling (NCM) & Hybrid Consistency Assistant
A parallel coarse-to-fine hierarchical neural code generator.
"""
import os
import sys
import argparse
import torch

from custom_ncm.config import NCMConfig
from custom_ncm.code_tokenizer import ByteCodeTokenizer
from custom_ncm.vqvae import CodeVQVAE
from custom_ncm.ncm_model import NextConceptGenerator
from custom_ncm.generate import generate_code_coarse_to_fine
from custom_ncm.hybrid_model import HybridConsistencyNCM

def get_param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def print_model_tiers():
    print("=" * 70)
    print("  CODE ROBOT: MODEL PARAMETER TIERS & ARCHITECTURE")
    print("=" * 70)
    
    tiers = [
        ("Small (Baseline)", NCMConfig.small()),
        ("Medium (Scaled)", NCMConfig.medium()),
        ("Large (Deep)", NCMConfig.large()),
        ("Option B (2.0 Billion Frontier Tier)", NCMConfig.two_billion())
    ]
    
    for name, cfg in tiers:
        with torch.device('meta'):
            v = CodeVQVAE(
                vocab_size=260,
                context_length=cfg.context_length,
                embed_dim=cfg.embed_dim,
                levels=cfg.levels,
                codebook_size=cfg.vq_codebook_size,
                n_enc_layers=cfg.vq_enc_layers,
                n_dec_layers=cfg.vq_dec_layers
            )
            g = NextConceptGenerator(
                vqvae=v,
                prefix_len=cfg.prefix_length,
                d_model=cfg.embed_dim,
                n_head=cfg.gen_n_head,
                n_layer=cfg.gen_n_layer,
                codebook_size=cfg.vq_codebook_size
            )
        v_params = get_param_count(v)
        g_params = get_param_count(g)
        tot_params = v_params + g_params
        
        print(f"\n* Tier: {name}")
        if tot_params >= 1e9:
            print(f"  - Total Parameters: {tot_params:,} (~{tot_params/1e9:.2f}B)")
        else:
            print(f"  - Total Parameters: {tot_params:,} (~{tot_params/1e6:.1f}M)")
        print(f"  - Concept VQ-VAE  : {v_params:,} params ({cfg.vq_enc_layers} enc / {cfg.vq_dec_layers} dec layers)")
        print(f"  - NCM Generator   : {g_params:,} params ({cfg.gen_n_layer} layers, {cfg.gen_n_head} heads, dim={cfg.embed_dim})")
        print(f"  - Context Scales  : target={cfg.context_length} tokens, prefix={cfg.prefix_length} tokens")
        print(f"  - Levels (Hier.)  : {cfg.levels}")
        
    print("\n* Tier: Hybrid Consistency Model (HC-NCM)")
    cfg = NCMConfig.small()
    v = CodeVQVAE(260, cfg.context_length, cfg.embed_dim, cfg.levels, cfg.vq_codebook_size, cfg.vq_enc_layers, cfg.vq_dec_layers)
    h = HybridConsistencyNCM(cfg, vqvae=v)
    h_params = get_param_count(h)
    print(f"  - Total Parameters: {h_params:,} (~{h_params/1e6:.1f}M)")
    print(f"  - LCM Consistency : 1-Step direct trajectory mapping (Diffusion Gemma bidirectional backbone)")
    print(f"  - NCM Hierarchy   : Coarse-to-fine multi-scale pyramid (2 -> 8 -> 32 -> 64 tokens)")
    print(f"  - Causal AR Head  : Gemma-style autoregressive syntax realizer")
    print(f"  - Compiler Loop   : Programmatic AST syntax verification")
    print("\n" + "=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Code Robot - Next-Concept Model Runner")
    parser.add_argument("--tiers", action="store_true", help="Display all parameter scaling tiers")
    parser.add_argument("--hybrid", action="store_true", help="Run in Hybrid Consistency Model mode (LCM + Gemma + NCM)")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt prefix for code generation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    args = parser.parse_args()
    
    if args.tiers:
        print_model_tiers()
        return

    print("=" * 70)
    print("  WELCOME TO CODE ROBOT (NEXT-CONCEPT & HYBRID CONSISTENCY MODELING)")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ByteCodeTokenizer()
    cfg = NCMConfig.small()
    v_ckpt = "checkpoints/vqvae_pos.pt" if os.path.exists("checkpoints/vqvae_pos.pt") else "checkpoints/vqvae.pt"
    
    if args.hybrid:
        print("[*] Running in HYBRID CONSISTENCY MODE (1-Step LCM + Bidirectional + NCM + Causal AR)")
        h_ckpt = "checkpoints/hybrid/hybrid_model.pt"
        if not os.path.exists(h_ckpt):
            print(f"[!] Hybrid checkpoint not found at {h_ckpt}. Training new hybrid checkpoint...")
            from custom_ncm.train_hybrid import train_hybrid_model
            hybrid_model = train_hybrid_model(config=cfg, epochs=3, device=device)
        else:
            v_data = torch.load(v_ckpt, map_location=device, weights_only=False)
            vqvae = CodeVQVAE(
                vocab_size=tokenizer.vocab_size,
                context_length=cfg.context_length,
                embed_dim=cfg.embed_dim,
                levels=cfg.levels,
                codebook_size=cfg.vq_codebook_size,
                n_enc_layers=cfg.vq_enc_layers,
                n_dec_layers=cfg.vq_dec_layers
            ).to(device)
            vqvae.load_state_dict(v_data["model"])
            vqvae.eval()
            
            hybrid_model = HybridConsistencyNCM(cfg, vqvae=vqvae).to(device)
            h_data = torch.load(h_ckpt, map_location=device, weights_only=False)
            hybrid_model.load_state_dict(h_data["model"], strict=False)
            hybrid_model.eval()
            
        prompt = args.prompt or "# Return the square of x\ndef square(x):\n"
        print(f"\n[+] Running Hybrid Generation on prompt:\n{prompt}")
        
        P = cfg.prefix_length
        pad_id = tokenizer.pad_token_id
        raw_ids = tokenizer.encode(prompt)
        p_ids = raw_ids[-P:] if len(raw_ids) > P else raw_ids + [pad_id] * (P - len(raw_ids))
        prefix_tensor = torch.tensor([p_ids], dtype=torch.long, device=device)
        
        res = hybrid_model.fast_generate(prefix_tensor, tokenizer)
        print(f"[*] Chosen Source: {res['chosen_source']}")
        print(f"[*] Generated Code:\n{res['full_code']}")
        print(f"[*] AST Valid: {res['is_valid_ast']}")
        return

    # Standard NCM Generator Mode
    ckpt_path = args.checkpoint or ("checkpoints/generator_official_python.pt" if os.path.exists("checkpoints/generator_official_python.pt") else "checkpoints/generator.pt")
        
    if os.path.exists(ckpt_path):
        print(f"[*] Loading Code Robot from: {ckpt_path}")
        v_data = torch.load(v_ckpt, map_location=device, weights_only=False)
        cfg = v_data.get("config", NCMConfig())
        
        vqvae = CodeVQVAE(
            vocab_size=tokenizer.vocab_size,
            context_length=cfg.context_length,
            embed_dim=cfg.embed_dim,
            levels=cfg.levels,
            codebook_size=cfg.vq_codebook_size,
            n_enc_layers=cfg.vq_enc_layers,
            n_dec_layers=cfg.vq_dec_layers
        ).to(device)
        vqvae.load_state_dict(v_data["model"])
        vqvae.eval()
        
        generator = NextConceptGenerator(
            vqvae=vqvae,
            prefix_len=cfg.prefix_length,
            d_model=cfg.embed_dim,
            n_head=cfg.gen_n_head,
            n_layer=cfg.gen_n_layer,
            codebook_size=cfg.vq_codebook_size
        ).to(device)
        g_data = torch.load(ckpt_path, map_location=device, weights_only=False)
        generator.load_state_dict(g_data["model"])
        generator.eval()
        
        prompt = args.prompt or "# Return the square of x\ndef square(x):\n"
        print(f"\n[+] Running generation on prompt:\n{prompt}")
        generate_code_coarse_to_fine(
            generator=generator,
            tokenizer=tokenizer,
            prompt=prompt,
            prefix_len=cfg.prefix_length,
            temperature=0.3,
            top_k=20,
            verbose=True
        )
    else:
        print("[!] No checkpoint found. Displaying model parameter tiers:")
        print_model_tiers()

if __name__ == "__main__":
    main()
