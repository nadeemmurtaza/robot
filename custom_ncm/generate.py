"""
Inference & Coarse-to-Fine Generation Visualizer for NCM Code Model.
Shows how code is refined level-by-level from a coarse concept to working code in parallel.
"""
import time
from typing import Optional, List, Tuple
import torch

from custom_ncm.config import NCMConfig
from custom_ncm.code_tokenizer import ByteCodeTokenizer
from custom_ncm.vqvae import CodeVQVAE
from custom_ncm.ncm_model import NextConceptGenerator
from custom_ncm.attention import convert_legacy_mha_state_dict

def load_ncm_model(
    vqvae_ckpt: str = "checkpoints/vqvae.pt",
    gen_ckpt: str = "checkpoints/generator.pt",
    device: Optional[torch.device] = None
) -> Tuple[NextConceptGenerator, ByteCodeTokenizer, NCMConfig]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    tokenizer = ByteCodeTokenizer()
    v_data = torch.load(vqvae_ckpt, map_location=device, weights_only=False)
    config = v_data.get("config", NCMConfig())
    
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
    vqvae.eval()

    g_data = torch.load(gen_ckpt, map_location=device, weights_only=False)
    g_config = g_data.get("config", config)
    # Detect number of layers from checkpoint
    n_layers = max([int(k.split(".")[1]) for k in g_data["model"].keys() if k.startswith("blocks.")], default=g_config.gen_n_layer - 1) + 1
    generator = NextConceptGenerator(
        vqvae=vqvae,
        prefix_len=g_config.prefix_length,
        d_model=g_config.embed_dim,
        n_head=g_config.gen_n_head,
        n_layer=n_layers,
        codebook_size=g_config.vq_codebook_size
    ).to(device)
    generator.load_state_dict(convert_legacy_mha_state_dict(g_data["model"]))
    generator.eval()
    
    return generator, tokenizer, config


def generate_code_coarse_to_fine(
    generator: NextConceptGenerator,
    tokenizer: ByteCodeTokenizer,
    prompt: str,
    prefix_len: int = 32,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    verbose: bool = True
) -> List[str]:
    """
    Takes a Python code prompt, performs parallel hierarchical generation,
    and returns the decoded text at each scale level.
    """
    device = next(generator.parameters()).device
    
    # Encode prompt
    raw_ids = tokenizer.encode(prompt)
    if len(raw_ids) > prefix_len:
        raw_ids = raw_ids[-prefix_len:]
    else:
        raw_ids = raw_ids + [tokenizer.pad_token_id] * (prefix_len - len(raw_ids))
        
    prefix_tensor = torch.tensor([raw_ids], dtype=torch.long, device=device)
    
    t0 = time.time()
    with torch.no_grad():
        codes, progression = generator.generate(
            prefix_tensor,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
    elapsed = time.time() - t0
    
    decoded_levels = []
    for l_idx, token_tensor in enumerate(progression):
        # token_tensor has shape [1, L]
        text = tokenizer.decode(token_tensor[0], skip_special_tokens=True)
        decoded_levels.append(text)
        
    if verbose:
        print("\n" + "="*70)
        print("  NEXT-CONCEPT MODELING (NCM) COARSE-TO-FINE CODE GENERATION")
        print("="*70)
        print(f"[*] Prompt (Prefix):\n{prompt}")
        print("-"*70)
        
        levels = generator.levels
        for l_idx, (l_len, text) in enumerate(zip(levels, decoded_levels)):
            print(f"\n[Level {l_idx}] (Scale: {l_len} parallel concepts):")
            safe_text = text.encode('ascii', errors='replace').decode('ascii')
            indented = "\n".join("   " + line for line in safe_text.splitlines())
            print(indented if indented.strip() else "   <empty / preliminary concept>")
            
        print("\n" + "-"*70)
        print(f"[+] Total parallel forward passes: {len(levels)} steps (vs {generator.vqvae.context_length} autoregressive steps)")
        speedup = generator.vqvae.context_length / len(levels)
        print(f"[+] Step reduction: {speedup:.1f}x fewer generation steps!")
        print(f"[+] Generation elapsed time: {elapsed*1000:.1f} ms")
        print("="*70 + "\n")
        
    return decoded_levels

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Python code coarse-to-fine using Next-Concept Modeling (NCM)")
    parser.add_argument("--prompt", type=str, default="", help="Prompt code string")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=30, help="Top-K filtering")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P nucleus filtering")
    parser.add_argument("--interactive", action="store_true", help="Interactive prompt loop")
    args = parser.parse_args()
    
    print("[*] Loading trained NCM models...")
    gen, tok, cfg = load_ncm_model()
    print("[+] Models loaded successfully!")
    
    if args.interactive:
        print("\n--- NCM INTERACTIVE CODE GENERATION (Type 'exit' to quit) ---")
        while True:
            try:
                user_p = input("\nEnter prompt prefix (e.g. 'def reverse_list(arr):'): ")
                if not user_p.strip() or user_p.strip() == "exit":
                    break
                generate_code_coarse_to_fine(
                    gen, tok, user_p + "\n",
                    temperature=args.temp,
                    top_k=args.top_k,
                    top_p=args.top_p
                )
            except KeyboardInterrupt:
                break
    else:
        prompt = args.prompt or "# Filter even numbers from list\ndef filter_evens(nums):\n"
        generate_code_coarse_to_fine(
            gen, tok, prompt,
            temperature=args.temp,
            top_k=args.top_k,
            top_p=args.top_p
        )

