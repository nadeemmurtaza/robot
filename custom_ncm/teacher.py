"""
Interactive Python Teacher & Curriculum Engine for Next-Concept Model.
Allows progressive chapter-by-chapter teaching as well as custom user lessons on the fly.
"""
import os
import sys
import time
from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from custom_ncm.config import NCMConfig
from custom_ncm.code_tokenizer import ByteCodeTokenizer
from custom_ncm.vqvae import CodeVQVAE
from custom_ncm.ncm_model import NextConceptGenerator
from custom_ncm.verifier import CodeVerifier, VerificationResult
from custom_ncm.curriculum import Lesson, get_curriculum, get_chapters

class PythonTeacher:
    """
    Teacher agent that instructs the NCM student model on Python concepts,
    tests its understanding, and provides immediate reinforcement feedback.
    """
    def __init__(
        self,
        config: Optional[NCMConfig] = None,
        vqvae_ckpt: str = "checkpoints/vqvae_pos.pt",
        gen_ckpt: str = "checkpoints/generator_base.pt",
        device: Optional[torch.device] = None
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        self.tokenizer = ByteCodeTokenizer()
        
        if config is None:
            v_data = torch.load(vqvae_ckpt, map_location=device, weights_only=False)
            self.config = v_data.get("config", NCMConfig())
        else:
            self.config = config
            
        # Load VQ-VAE
        v_data = torch.load(vqvae_ckpt, map_location=device, weights_only=False)
        self.vqvae = CodeVQVAE(
            vocab_size=self.tokenizer.vocab_size,
            context_length=self.config.context_length,
            embed_dim=self.config.embed_dim,
            levels=self.config.levels,
            codebook_size=self.config.vq_codebook_size,
            n_enc_layers=self.config.vq_enc_layers,
            n_dec_layers=self.config.vq_dec_layers
        ).to(device)
        self.vqvae.load_state_dict(v_data["model"])
        self.vqvae.eval()
        for p in self.vqvae.parameters():
            p.requires_grad = False
            
        # Load Generator
        g_data = torch.load(gen_ckpt, map_location=device, weights_only=False)
        self.generator = NextConceptGenerator(
            vqvae=self.vqvae,
            prefix_len=self.config.prefix_length,
            d_model=self.config.embed_dim,
            n_head=self.config.gen_n_head,
            n_layer=self.config.gen_n_layer,
            codebook_size=self.config.vq_codebook_size
        ).to(device)
        self.generator.load_state_dict(g_data["model"])
        
        self.optimizer = AdamW(
            self.generator.parameters(),
            lr=self.config.lr_gen * 0.5,
            weight_decay=self.config.weight_decay
        )

    def test_student(self, prompt: str, func_name: str, test_cases: List[Tuple[Any, Any]]) -> Tuple[str, VerificationResult]:
        """Runs the student on a prompt and verifies the code."""
        self.generator.eval()
        P = self.config.prefix_length
        pad_id = self.tokenizer.pad_token_id
        
        raw_ids = self.tokenizer.encode(prompt)
        p_ids = raw_ids[-P:] if len(raw_ids) > P else raw_ids + [pad_id] * (P - len(raw_ids))
        prefix_tensor = torch.tensor([p_ids], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            _, progression = self.generator.generate(
                prefix_tensor,
                temperature=0.2, # Low temperature for testing accuracy
                top_k=10,
                top_p=0.9
            )
            
        final_tokens = progression[-1][0]
        generated_body = self.tokenizer.decode(final_tokens, skip_special_tokens=True)
        full_code = prompt + generated_body
        
        v_res = CodeVerifier.run_tests(full_code, func_name, test_cases)
        return full_code, v_res

    def teach_lesson(self, lesson: Lesson, teaching_steps: int = 8, verbose: bool = True) -> bool:
        """
        Teaches a single Python lesson:
        1. Tests student's initial attempt.
        2. Applies concept reinforcement steps on the correct lesson implementation.
        3. Tests student's post-teaching attempt to verify mastery.
        """
        if verbose:
            print("\n" + "="*70, flush=True)
            print(f"  TEACHING LESSON: {lesson.topic.upper()}", flush=True)
            print(f"  Chapter: {lesson.chapter}", flush=True)
            print("="*70, flush=True)
            print(f"Prompt:\n{lesson.prompt.strip()}", flush=True)
            print(f"Expected Target:\n{lesson.target_code.strip()}", flush=True)
            print("-"*70, flush=True)
            
        # Initial Student Attempt
        before_code, before_res = self.test_student(lesson.prompt, lesson.func_name, lesson.test_cases)
        if verbose:
            print("[Before Teaching Attempt]:", flush=True)
            for line in before_code.splitlines():
                print(f"   {line}", flush=True)
            print(f"   Tests Passed: {before_res.tests_passed}/{before_res.total_tests} | Syntax Valid: {before_res.is_valid_syntax}", flush=True)
            
        if before_res.passes_unit_tests:
            if verbose:
                print("[MASTERED] Student already mastered this lesson! No correction needed.", flush=True)
            return True
            
        # Teaching & Reinforcement Step
        if verbose:
            print(f"\n[*] Applying {teaching_steps} concept instruction steps...", flush=True)
            
        P = self.config.prefix_length
        L = self.config.context_length
        pad_id = self.tokenizer.pad_token_id
        
        p_ids = self.tokenizer.encode(lesson.prompt)
        p_ids = p_ids[-P:] if len(p_ids) > P else p_ids + [pad_id] * (P - len(p_ids))
        
        t_ids = self.tokenizer.encode(lesson.target_code)
        t_ids = t_ids[:L] if len(t_ids) > L else t_ids + [pad_id] * (L - len(t_ids))
        
        prefix_t = torch.tensor([p_ids], dtype=torch.long, device=self.device)
        target_t = torch.tensor([t_ids], dtype=torch.long, device=self.device)
        
        self.generator.train()
        for step in range(teaching_steps):
            self.optimizer.zero_grad()
            _, _, loss = self.generator(prefix_t, target_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
            self.optimizer.step()
            
        # Post-Teaching Student Attempt
        after_code, after_res = self.test_student(lesson.prompt, lesson.func_name, lesson.test_cases)
        if verbose:
            print("\n[After Teaching Attempt]:", flush=True)
            for line in after_code.splitlines():
                print(f"   {line}", flush=True)
            print(f"   Tests Passed: {after_res.tests_passed}/{after_res.total_tests} | Syntax Valid: {after_res.is_valid_syntax}", flush=True)
            
            if after_res.passes_unit_tests:
                print(f"[SUCCESS] Student successfully learned {lesson.topic}! 100% Error-Free!", flush=True)
            else:
                print(f"[PROGRESS] Score: {after_res.score:.2f} | Syntax: {after_res.is_valid_syntax}", flush=True)
            print("="*70 + "\n", flush=True)
            
        return after_res.passes_unit_tests

    def teach_chapter(self, chapter_name: str):
        """Teaches an entire curriculum chapter."""
        chapters = get_chapters()
        if chapter_name not in chapters:
            print(f"Chapter '{chapter_name}' not found. Available: {list(chapters.keys())}")
            return
            
        lessons = chapters[chapter_name]
        print("\n" + "#"*70, flush=True)
        print(f"  COMMENCING {chapter_name.upper()} ({len(lessons)} Lessons)", flush=True)
        print("#"*70, flush=True)
        
        passed_count = 0
        for lesson in lessons:
            ok = self.teach_lesson(lesson)
            if ok:
                passed_count += 1
                
        print(f"\n[Chapter Summary] Mastered {passed_count}/{len(lessons)} lessons in {chapter_name}!", flush=True)

    def teach_custom(self, prompt: str, target_body: str, func_name: str, test_cases: List[Tuple[Any, Any]], steps: int = 5):
        """Allows the user to teach any custom Python function on the fly."""
        custom_lesson = Lesson(
            chapter="User Custom Lesson",
            topic=func_name,
            prompt=prompt if prompt.endswith("\n") else prompt + "\n",
            func_name=func_name,
            target_code=target_body if target_body.endswith("\n") else target_body + "\n",
            test_cases=test_cases
        )
        return self.teach_lesson(custom_lesson, teaching_steps=steps)
