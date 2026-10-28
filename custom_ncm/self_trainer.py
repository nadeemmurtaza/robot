"""
Self-Training Engine for Error-Free Code Generation.
Uses AST validation, unit-test execution feedback, and rejection fine-tuning (Expert Iteration)
to autonomously train the Next-Concept Model until it generates error-free, verifiable code.
"""
import os
import sys
import time
from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from custom_ncm.config import NCMConfig
from custom_ncm.code_tokenizer import ByteCodeTokenizer
from custom_ncm.dataset import CodeDataset
from custom_ncm.vqvae import CodeVQVAE
from custom_ncm.ncm_model import NextConceptGenerator
from custom_ncm.verifier import CodeVerifier, VerificationResult
from custom_ncm.task_bank import CodingTask, get_all_tasks

class SelfTrainingEngine:
    """
    Autonomous engine that:
    1. Generates code solutions from the NCM model.
    2. Verifies code using AST syntax parsing and unit-test execution.
    3. Curates a verified dataset of 100% error-free samples.
    4. Automatically fine-tunes the model on verified solutions.
    5. Repeats until syntax validity and test pass rates reach target goals.
    """
    def __init__(
        self,
        config: NCMConfig,
        vqvae: CodeVQVAE,
        generator: NextConceptGenerator,
        tokenizer: ByteCodeTokenizer,
        checkpoint_dir: str = "checkpoints/self_trained"
    ):
        self.config = config
        self.vqvae = vqvae
        self.generator = generator
        self.tokenizer = tokenizer
        self.checkpoint_dir = checkpoint_dir
        self.device = next(generator.parameters()).device
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.tasks = get_all_tasks()
        self.history: List[Dict[str, Any]] = []

    def evaluate_task_bank(
        self,
        samples_per_task: int = 4,
        temperature: float = 0.7
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Runs generation for all tasks in the bank and verifies them.
        Returns detailed results and summary statistics.
        """
        self.generator.eval()
        results = []
        total_evals = 0
        syntax_valid_count = 0
        tests_passed_count = 0
        total_score = 0.0
        
        P = self.config.prefix_length
        pad_id = self.tokenizer.pad_token_id
        
        for task in self.tasks:
            raw_ids = self.tokenizer.encode(task.prompt)
            if len(raw_ids) > P:
                p_ids = raw_ids[-P:]
            else:
                p_ids = raw_ids + [pad_id] * (P - len(raw_ids))
                
            prefix_tensor = torch.tensor([p_ids], dtype=torch.long, device=self.device)
            
            task_candidates = []
            for _ in range(samples_per_task):
                total_evals += 1
                with torch.no_grad():
                    _, progression = self.generator.generate(
                        prefix_tensor,
                        temperature=temperature,
                        top_k=30,
                        top_p=0.9
                    )
                # Final level output (Level 3)
                final_tokens = progression[-1][0]
                generated_body = self.tokenizer.decode(final_tokens, skip_special_tokens=True)
                full_code = task.prompt + generated_body
                
                # Verify code
                v_res = CodeVerifier.run_tests(full_code, task.func_name, task.test_cases)
                if v_res.is_valid_syntax:
                    syntax_valid_count += 1
                if v_res.passes_unit_tests:
                    tests_passed_count += 1
                total_score += v_res.score
                
                task_candidates.append({
                    "generated_body": generated_body,
                    "full_code": full_code,
                    "verification": v_res
                })
                
            results.append({
                "task": task,
                "candidates": task_candidates
            })
            
        stats = {
            "total_samples": total_evals,
            "syntax_valid_rate": (syntax_valid_count / max(total_evals, 1)) * 100.0,
            "pass_rate": (tests_passed_count / max(total_evals, 1)) * 100.0,
            "mean_score": total_score / max(total_evals, 1)
        }
        return results, stats

    def run_self_training_loop(
        self,
        iterations: int = 5,
        samples_per_task: int = 4,
        fine_tune_epochs_per_round: int = 4,
        target_pass_rate: float = 95.0
    ):
        """
        Autonomous Self-Training Execution Loop:
        1. Evaluate current policy.
        2. Filter and collect error-free passing completions.
        3. Augment with reference exemplars for missed tasks.
        4. Fine-tune Next-Concept Generator on verified error-free data.
        5. Checkpoint & progress report.
        """
        print("\n" + "="*70, flush=True)
        print("  STARTING AUTONOMOUS SELF-TRAINING ENGINE (NCM FOR ERROR-FREE CODE)", flush=True)
        print("="*70, flush=True)
        print(f"[*] Total Problem Bank Tasks: {len(self.tasks)}", flush=True)
        print(f"[*] Iterations Planned: {iterations}", flush=True)
        print(f"[*] Samples per task per round: {samples_per_task}", flush=True)
        print(f"[*] Target Pass Rate: {target_pass_rate}%\n", flush=True)
        
        optimizer = AdamW(
            self.generator.parameters(),
            lr=self.config.lr_gen * 0.7,
            weight_decay=self.config.weight_decay
        )
        
        best_pass_rate = -1.0
        
        for round_idx in range(1, iterations + 1):
            t0 = time.time()
            print(f"\n--- [Round {round_idx}/{iterations}] Evaluating & Generating Self-Samples ---", flush=True)
            
            # Step 1: Rollout & Verify
            results, stats = self.evaluate_task_bank(samples_per_task=samples_per_task)
            
            print(f"[*] Syntax Validity Rate : {stats['syntax_valid_rate']:.1f}%", flush=True)
            print(f"[*] Unit-Test Pass Rate  : {stats['pass_rate']:.1f}%", flush=True)
            print(f"[*] Average Quality Score: {stats['mean_score']:.3f}", flush=True)
            
            # Step 2: Build Verified Error-Free Dataset
            verified_pairs: List[Tuple[str, str]] = []
            
            for item in results:
                task = item["task"]
                passed_candidates = [
                    c for c in item["candidates"] if c["verification"].passes_unit_tests
                ]
                if passed_candidates:
                    # Model successfully generated error-free solutions on its own!
                    for c in passed_candidates:
                        verified_pairs.append((task.prompt, c["generated_body"]))
                else:
                    # Model struggled on this task; feed ground-truth reference to teach it
                    verified_pairs.append((task.prompt, task.reference_body))
                    
            print(f"[+] Curated {len(verified_pairs)} verified error-free solutions for fine-tuning.", flush=True)
            
            # Step 3: Self-Supervised Rejection Fine-Tuning Step
            train_ds = CodeDataset(
                verified_pairs,
                self.tokenizer,
                self.config.prefix_length,
                self.config.context_length
            )
            train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
            
            self.generator.train()
            total_train_loss = 0.0
            for epoch in range(1, fine_tune_epochs_per_round + 1):
                for batch in train_loader:
                    prefix = batch["prefix"].to(self.device)
                    target = batch["target"].to(self.device)
                    
                    optimizer.zero_grad()
                    _, _, loss = self.generator(prefix, target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.config.grad_clip)
                    optimizer.step()
                    total_train_loss += loss.item()
                    
            round_elapsed = time.time() - t0
            print(f"[+] Round {round_idx} self-training completed in {round_elapsed:.1f}s.", flush=True)
            
            # Record metrics
            self.history.append({
                "round": round_idx,
                "stats": stats,
                "dataset_size": len(verified_pairs)
            })
            
            # Save Checkpoint if improved
            if stats["pass_rate"] > best_pass_rate or round_idx == iterations:
                best_pass_rate = max(best_pass_rate, stats["pass_rate"])
                save_path = os.path.join(self.checkpoint_dir, "best_error_free_generator.pt")
                torch.save({
                    "model": self.generator.state_dict(),
                    "config": self.config,
                    "round": round_idx,
                    "stats": stats
                }, save_path)
                print(f"[+] Saved checkpoint to {save_path}", flush=True)
                
            if stats["pass_rate"] >= target_pass_rate and stats["syntax_valid_rate"] >= 98.0:
                print(f"\n[🎉] Target accuracy achieved! Error-free code generation goal met at round {round_idx}!", flush=True)
                break
                
        # Final Verification
        print("\n" + "="*70, flush=True)
        print("  FINAL EVALUATION OF SELF-TRAINED MODEL", flush=True)
        print("="*70, flush=True)
        final_results, final_stats = self.evaluate_task_bank(samples_per_task=2, temperature=0.2)
        print(f"[*] Final Syntax Validity Rate : {final_stats['syntax_valid_rate']:.1f}%", flush=True)
        print(f"[*] Final Unit-Test Pass Rate  : {final_stats['pass_rate']:.1f}%", flush=True)
        print(f"[*] Final Average Quality Score: {final_stats['mean_score']:.3f}", flush=True)
        print("="*70 + "\n", flush=True)
        return final_stats
