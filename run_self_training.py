"""
Autonomous Self-Training Pipeline for Error-Free Code Generation.
1. Trains / Verifies Stage 1 VQ-VAE (with position-aware multiscale decoder).
2. Trains Base Stage 2 Next-Concept Generator.
3. Launches the Self-Training Engine to autonomously evaluate, filter,
   and self-optimize on verified error-free solutions.
"""
import os
import sys
import torch

# Maximize CPU parallelism
torch.set_num_threads(8)

from custom_ncm.config import NCMConfig
from custom_ncm.code_tokenizer import ByteCodeTokenizer
from custom_ncm.train_vqvae import train_vqvae
from custom_ncm.train_ncm import train_ncm
from custom_ncm.self_trainer import SelfTrainingEngine
from custom_ncm.generate import generate_code_coarse_to_fine
from custom_ncm.verifier import CodeVerifier
from custom_ncm.task_bank import get_all_tasks

def main():
    print("=" * 70, flush=True)
    print("  AUTONOMOUS SELF-TRAINING ENGINE: NCM FOR ERROR-FREE CODING", flush=True)
    print("=" * 70, flush=True)
    
    config = NCMConfig(
        context_length=64,
        prefix_length=32,
        levels=[2, 8, 32, 64],
        embed_dim=128,
        vq_codebook_size=256,
        vq_enc_layers=2,
        vq_dec_layers=2,
        gen_n_layer=3,
        gen_n_head=4,
        batch_size=16,
        lr_vqvae=1.5e-3,
        lr_gen=1e-3
    )
    
    # Step 1: Train VQ-VAE with position-aware decoder
    print("\n[Step 1] Initializing and training position-aware VQ-VAE...", flush=True)
    vqvae_ckpt = "checkpoints/vqvae_pos.pt"
    vqvae = train_vqvae(config, epochs=8, checkpoint_path=vqvae_ckpt)
    
    # Step 2: Train initial Next-Concept Generator
    print("\n[Step 2] Initializing and training base Next-Concept Generator...", flush=True)
    gen_ckpt = "checkpoints/generator_base.pt"
    generator = train_ncm(config, vqvae=vqvae, gen_checkpoint=gen_ckpt, epochs=10)
    
    # Step 3: Launch Autonomous Self-Training Engine
    print("\n[Step 3] Launching Autonomous Self-Training Engine...", flush=True)
    tokenizer = ByteCodeTokenizer()
    engine = SelfTrainingEngine(
        config=config,
        vqvae=vqvae,
        generator=generator,
        tokenizer=tokenizer,
        checkpoint_dir="checkpoints/self_trained"
    )
    
    # Run 4 self-training rounds
    final_stats = engine.run_self_training_loop(
        iterations=4,
        samples_per_task=3,
        fine_tune_epochs_per_round=4,
        target_pass_rate=95.0
    )
    
    # Step 4: Verification Showcase
    print("\n" + "=" * 70, flush=True)
    print("  VERIFYING SELF-TRAINED CODE COMPLETED TASKS", flush=True)
    print("=" * 70, flush=True)
    
    tasks = get_all_tasks()[:4]
    for task in tasks:
        print(f"\n[Task: {task.task_id}] {task.func_name}")
        progression = generate_code_coarse_to_fine(
            generator=generator,
            tokenizer=tokenizer,
            prompt=task.prompt,
            prefix_len=config.prefix_length,
            temperature=0.3,
            top_k=20,
            verbose=False
        )
        full_code = task.prompt + progression[-1]
        v_res = CodeVerifier.run_tests(full_code, task.func_name, task.test_cases)
        
        print("Generated Code:")
        for line in full_code.splitlines():
            print(f"   {line}")
        print(f"AST Syntax Valid: {v_res.is_valid_syntax}")
        print(f"Unit Tests Pass : {v_res.passes_unit_tests} ({v_res.tests_passed}/{v_res.total_tests} passed)")
        if v_res.error_message:
            print(f"Diagnostics     : {v_res.error_type} - {v_res.error_message}")
        print("-" * 50)

if __name__ == "__main__":
    main()
