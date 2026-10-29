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

from custom_ncm.config import NCMConfig, add_tier_cli_arg
from custom_ncm.code_tokenizer import ByteCodeTokenizer
from custom_ncm.dataset import CodeDataset
from custom_ncm.vqvae import CodeVQVAE
from custom_ncm.ncm_model import NextConceptGenerator
from custom_ncm.generate import generate_code_coarse_to_fine
from custom_ncm.official_python_harvester import OfficialPythonHarvester
from custom_ncm.train_vqvae import get_device, _warn_if_hardware_insufficient
from custom_ncm.verifier import CodeVerifier
from custom_ncm.attention import convert_legacy_mha_state_dict

def evaluate_on_held_out(
    generator: NextConceptGenerator,
    tokenizer: ByteCodeTokenizer,
    config: NCMConfig,
    test_file: str = "data/test.jsonl",
    max_samples: int = 100
) -> dict:
    """
    Real correctness eval instead of bare cross-entropy loss: generates a
    completion for each held-out prompt and checks with CodeVerifier whether
    the result is even syntactically valid Python, plus whether it matches
    the ground-truth body exactly. (No per-function unit tests exist in this
    harvested stdlib corpus, so functional pass@k isn't available here - this
    at least tells you whether the model outputs real code or noise.)
    """
    if not os.path.exists(test_file):
        print(f"  [!] No held-out test file at {test_file}, skipping correctness eval.")
        return {}

    samples = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    samples = samples[:max_samples]

    syntax_valid = 0
    exact_match = 0
    generator.eval()
    for item in samples:
        prompt, gt_body = item["prompt"], item["body"]
        decoded_levels = generate_code_coarse_to_fine(
            generator=generator, tokenizer=tokenizer, prompt=prompt,
            prefix_len=config.prefix_length, temperature=0.0, verbose=False
        )
        gen_body = decoded_levels[-1]  # finest level = final generated code
        full_code = prompt + gen_body
        ok, _, _ = CodeVerifier.check_syntax(full_code)
        syntax_valid += int(ok)
        exact_match += int(gen_body.strip() == gt_body.strip())

    n = max(len(samples), 1)
    results = {
        "n_samples": len(samples),
        "syntax_valid_pct": 100.0 * syntax_valid / n,
        "exact_match_pct": 100.0 * exact_match / n,
    }
    print(
        f"[*] Held-out eval ({len(samples)} samples): "
        f"syntax-valid {results['syntax_valid_pct']:.1f}% | "
        f"exact-match {results['exact_match_pct']:.1f}%", flush=True
    )
    return results

def main(
    epochs: int = 8,
    checkpoint_path: str = "checkpoints/generator_official_python.pt",
    tier: str = "small"
):
    config_overrides = NCMConfig.from_tier(tier)
    print("=" * 70, flush=True)
    print("  TRAINING NCM MODEL ON OFFICIAL PYTHON STANDARD LIBRARY & DOCS", flush=True)
    print("=" * 70, flush=True)
    
    corpus_file = "data/official_python_corpus.json"
    train_split_file = "data/train.jsonl"

    if os.path.exists(train_split_file):
        # Train on the 80% split only - official_python_corpus.json is the
        # superset split_dataset.py carved train/val/test.jsonl out of, so
        # training on all of it would leak the held-out test set.
        corpus = []
        with open(train_split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    corpus.append(json.loads(line))
        print(f"[*] Loaded {len(corpus)} functions from train split ({train_split_file}).", flush=True)
    else:
        if not os.path.exists(corpus_file):
            harvester = OfficialPythonHarvester()
            corpus = harvester.harvest_all(corpus_file)
        else:
            with open(corpus_file, "r", encoding="utf-8") as f:
                corpus = json.load(f)
        print(
            f"[!] No {train_split_file} found - training on the FULL corpus "
            f"({len(corpus)} functions). Run split_dataset.py first to get a "
            "proper held-out test set and avoid leakage in the eval below.",
            flush=True
        )

    # Extract pairs
    pairs = [(item["prompt"], item["body"]) for item in corpus]
    
    tokenizer = ByteCodeTokenizer()
    config = config_overrides

    device = get_device()
    print(f"[*] Training on device: {device} (tier: {tier})", flush=True)
    _warn_if_hardware_insufficient(config, device)

    # Checkpoint names are tier-suffixed: a small-tier vqvae/generator has
    # different shapes than the 2B/1M-context one, so sharing one filename
    # across tiers would load mismatched weights (or crash on shape mismatch).
    tier_suffix = "" if tier == "small" else f"_{tier}"
    vqvae_ckpt = f"checkpoints/vqvae_pos{tier_suffix}.pt"
    gen_ckpt = f"checkpoints/generator_base{tier_suffix}.pt"
    if tier_suffix and not os.path.exists(vqvae_ckpt) and os.path.exists("checkpoints/vqvae_pos.pt"):
        vqvae_ckpt = "checkpoints/vqvae_pos.pt"  # legacy default for backward compat

    # Load VQ-VAE
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
            n_dec_layers=config.vq_dec_layers,
            attn_window=config.attn_window,
            attn_num_global=config.attn_num_global
        ).to(device)
        vqvae.load_state_dict(convert_legacy_mha_state_dict(v_data["model"]))

    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False

    # Load Base Generator
    generator = NextConceptGenerator(
        vqvae=vqvae,
        prefix_len=config.prefix_length,
        d_model=config.embed_dim,
        n_head=config.gen_n_head,
        n_layer=config.gen_n_layer,
        codebook_size=config.vq_codebook_size,
        attn_window=config.attn_window,
        attn_num_global=config.attn_num_global
    ).to(device)

    if os.path.exists(gen_ckpt):
        g_data = torch.load(gen_ckpt, map_location=device, weights_only=False)
        generator.load_state_dict(convert_legacy_mha_state_dict(g_data["model"]), strict=False)
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

    # Real correctness eval on the held-out test split (not just loss)
    print("=" * 70, flush=True)
    print("  HELD-OUT CORRECTNESS EVAL (data/test.jsonl)", flush=True)
    print("=" * 70, flush=True)
    evaluate_on_held_out(generator, tokenizer, config)

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
    import argparse
    parser = argparse.ArgumentParser(description="Train NCM on the official Python stdlib corpus")
    add_tier_cli_arg(parser)
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()
    tier_suffix = "" if args.tier == "small" else f"_{args.tier}"
    main(
        epochs=args.epochs,
        checkpoint_path=f"checkpoints/generator_official_python{tier_suffix}.pt",
        tier=args.tier
    )
