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
from custom_ncm.attention import convert_legacy_mha_state_dict
from custom_ncm.train_vqvae import get_device, _warn_if_hardware_insufficient

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
        ("Frontier Tier (2.04B Parameters / 1M Context)", NCMConfig.two_billion_1m())
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
        print(f"  - Context Window  : {cfg.context_length:,} tokens (prefix={cfg.prefix_length:,} tokens)")
        print(f"  - Levels (Hier.)  : {cfg.levels}")
        if cfg.context_length > 8192:
            print(f"  - Position Scheme : RoPE (Rotary Position Embeddings, theta=10,000,000, 0 parameter table)")
            print(f"  - TPU v5e-1 Memory: ~4.09 GB in bfloat16 (fits in 16 GB HBM2 with 11.9 GB headroom)")
        
    print("\n" + "-" * 70)
    print("  HYBRID CONSISTENCY ARCHITECTURE (HC-NCM): 1-Step LCM + Gemma + NCM")
    print("-" * 70)
    
    # 1. Baseline HC-NCM
    cfg_small = NCMConfig.small()
    with torch.device('meta'):
        v_s = CodeVQVAE(260, cfg_small.context_length, cfg_small.embed_dim, cfg_small.levels, cfg_small.vq_codebook_size, cfg_small.vq_enc_layers, cfg_small.vq_dec_layers)
        h_s = HybridConsistencyNCM(cfg_small, vqvae=v_s)
    h_s_params = get_param_count(h_s)
    print(f"\n* HC-NCM Baseline (CPU / Fast Exploration):")
    print(f"  - Total Parameters: {h_s_params:,} (~{h_s_params/1e6:.1f}M)")
    print(f"  - Context Window  : {cfg_small.context_length} tokens (prefix={cfg_small.prefix_length} tokens)")
    print(f"  - Hierarchy Levels: {cfg_small.levels}")
    
    # 2. Frontier HC-NCM 2.04B with 1M Context
    cfg_2b = NCMConfig.two_billion_1m()
    with torch.device('meta'):
        v_2b = CodeVQVAE(260, cfg_2b.context_length, cfg_2b.embed_dim, cfg_2b.levels, cfg_2b.vq_codebook_size, cfg_2b.vq_enc_layers, cfg_2b.vq_dec_layers)
        h_2b = HybridConsistencyNCM(cfg_2b, vqvae=v_2b)
    h_2b_params = get_param_count(h_2b)
    print(f"\n* HC-NCM Frontier (2.0B+ / 1M Context for TPU v5e-1 & 40GB RAM):")
    print(f"  - Total Parameters: {h_2b_params:,} (~{h_2b_params/1e9:.2f}B)")
    print(f"  - Context Window  : {cfg_2b.context_length:,} tokens (1,048,576 tokens / 1M)")
    print(f"  - Prompt Prefix   : {cfg_2b.prefix_length:,} tokens (131,072 tokens / 128k)")
    print(f"  - Hierarchy Levels: {cfg_2b.levels} (Level 0 gives 16,384x global compression)")
    print(f"  - LCM Consistency : 1-Step direct trajectory mapping (Diffusion Gemma bidirectional backbone)")
    print(f"  - Position Scheme : RoPE (Rotary Position Embeddings, theta=10,000,000, 0 parameter table)")
    print(f"  - Causal AR Head  : Gemma-style autoregressive syntax realizer")
    print(f"  - Hardware Sizing : ~4.09 GB in bfloat16 on TPU v5e-1 (16GB HBM2) / ~32GB training on 40GB RAM")
    print(f"  - AST Verification: Programmatic grammar and compiler execution loop")
    print("\n" + "=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Code Robot - Next-Concept Model Runner")
    parser.add_argument("--tiers", action="store_true", help="Display all parameter scaling tiers")
    parser.add_argument("--hybrid", action="store_true", help="Run in Hybrid Consistency Model mode (LCM + Gemma + NCM)")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt prefix for code generation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    from custom_ncm.config import add_tier_cli_arg
    add_tier_cli_arg(parser)
    args = parser.parse_args()

    if args.tiers:
        print_model_tiers()
        return

    print("=" * 70)
    print("  WELCOME TO CODE ROBOT (NEXT-CONCEPT & HYBRID CONSISTENCY MODELING)")
    print("=" * 70)

    device = get_device()
    tokenizer = ByteCodeTokenizer()
    cfg = NCMConfig.from_tier(args.tier)
    _warn_if_hardware_insufficient(cfg, device)
    tier_suffix = "" if args.tier == "small" else f"_{args.tier}"
    v_ckpt_candidates = [f"checkpoints/vqvae_pos{tier_suffix}.pt", f"checkpoints/vqvae{tier_suffix}.pt"]
    if tier_suffix:
        v_ckpt_candidates += ["checkpoints/vqvae_pos.pt", "checkpoints/vqvae.pt"]  # legacy fallback
    v_ckpt = next((p for p in v_ckpt_candidates if os.path.exists(p)), v_ckpt_candidates[0])

    if args.hybrid:
        print(f"[*] Running in HYBRID CONSISTENCY MODE (1-Step LCM + Bidirectional + NCM + Causal AR) - tier: {args.tier}")
        if cfg.context_length > 8192:
            print(
                "  [*] Note: at this tier, hybrid cross-attention only guarantees the LCM concept "
                "blueprint is fully visible (bounded global budget), not the full prompt prefix - "
                "see custom_ncm/hybrid_model.py's causal_blocks comment."
            )
        h_ckpt = f"checkpoints/hybrid/hybrid_model{tier_suffix}.pt"
        if not os.path.exists(h_ckpt):
            print(f"[!] Hybrid checkpoint not found at {h_ckpt}. Training new hybrid checkpoint...")
            from custom_ncm.train_hybrid import train_hybrid_model
            hybrid_model = train_hybrid_model(config=cfg, tier=args.tier, epochs=3, device=device)
        else:
            v_data = torch.load(v_ckpt, map_location=device, weights_only=False)
            vqvae = CodeVQVAE(
                vocab_size=tokenizer.vocab_size,
                context_length=cfg.context_length,
                embed_dim=cfg.embed_dim,
                levels=cfg.levels,
                codebook_size=cfg.vq_codebook_size,
                n_enc_layers=cfg.vq_enc_layers,
                n_dec_layers=cfg.vq_dec_layers,
                attn_window=cfg.attn_window,
                attn_num_global=cfg.attn_num_global
            ).to(device)
            vqvae.load_state_dict(convert_legacy_mha_state_dict(v_data["model"]))
            vqvae.eval()

            hybrid_model = HybridConsistencyNCM(cfg, vqvae=vqvae).to(device)
            h_data = torch.load(h_ckpt, map_location=device, weights_only=False)
            hybrid_model.load_state_dict(convert_legacy_mha_state_dict(h_data["model"]), strict=False)
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
    ckpt_path = args.checkpoint or (
        f"checkpoints/generator_official_python{tier_suffix}.pt"
        if os.path.exists(f"checkpoints/generator_official_python{tier_suffix}.pt")
        else f"checkpoints/generator{tier_suffix}.pt"
    )

    if os.path.exists(ckpt_path):
        print(f"[*] Loading Code Robot from: {ckpt_path}")
        v_data = torch.load(v_ckpt, map_location=device, weights_only=False)
        cfg = v_data.get("config", cfg)

        vqvae = CodeVQVAE(
            vocab_size=tokenizer.vocab_size,
            context_length=cfg.context_length,
            embed_dim=cfg.embed_dim,
            levels=cfg.levels,
            codebook_size=cfg.vq_codebook_size,
            n_enc_layers=cfg.vq_enc_layers,
            n_dec_layers=cfg.vq_dec_layers,
            attn_window=cfg.attn_window,
            attn_num_global=cfg.attn_num_global
        ).to(device)
        vqvae.load_state_dict(convert_legacy_mha_state_dict(v_data["model"]))
        vqvae.eval()

        g_data = torch.load(ckpt_path, map_location=device, weights_only=False)
        # The generator checkpoint's own layer count can differ from cfg.gen_n_layer
        # (cfg here comes from the VQ-VAE checkpoint, not the generator one) - detect
        # it from the checkpoint's keys instead of assuming, same fix as generate.py.
        n_layers = max(
            [int(k.split(".")[1]) for k in g_data["model"].keys() if k.startswith("blocks.")],
            default=cfg.gen_n_layer - 1
        ) + 1
        generator = NextConceptGenerator(
            vqvae=vqvae,
            prefix_len=cfg.prefix_length,
            d_model=cfg.embed_dim,
            n_head=cfg.gen_n_head,
            n_layer=n_layers,
            codebook_size=cfg.vq_codebook_size,
            attn_window=cfg.attn_window,
            attn_num_global=cfg.attn_num_global
        ).to(device)
        generator.load_state_dict(convert_legacy_mha_state_dict(g_data["model"]))
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
