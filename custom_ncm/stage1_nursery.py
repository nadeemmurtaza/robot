"""
Stage 1: The "Goo-Goo Ga-Ga" Nursery Phase for the AI Baby.
Teaches fundamental code phonics:
1. Letters, Digits, and Whitespace
2. Keyword Phonics (def, return, if, else, for, in, True, False, None)
3. Matching Bracket Pairs ((), [], {})
4. Atomic 1-line Statements (x = 1, return x, pass)
"""
import os
import random
import time
from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from custom_ncm.code_tokenizer import ByteCodeTokenizer
from custom_ncm.config import NCMConfig
from custom_ncm.vqvae import CodeVQVAE
from custom_ncm.ncm_model import NextConceptGenerator

# The "Baby Primer" - Basic Python Phonics and Syllables
PHONICS_PRIMER: List[Tuple[str, str]] = [
    # Category 1: Single Keywords & Syllables
    ("de", "f \n"),
    ("retu", "rn \n"),
    ("whi", "le \n"),
    ("fo", "r \n"),
    ("eli", "f \n"),
    ("els", "e:\n"),
    ("imp", "ort \n"),
    ("pri", "nt\n"),
    ("cla", "ss \n"),
    ("pas", "s\n"),
    ("Tr", "ue\n"),
    ("Fal", "se\n"),
    ("No", "ne\n"),
    
    # Category 2: Bracket & Punctuation Matching
    ("(", ")\n"),
    ("[", "]\n"),
    ("{", "}\n"),
    ("\"", "\"\n"),
    ("'", "'\n"),
    ("foo(", ")\n"),
    ("items[", "]\n"),
    ("data{", "}\n"),
    ("print(", "'hello')\n"),
    
    # Category 3: Atomic Assignments & Variables
    ("x = ", "0\n"),
    ("y = ", "1\n"),
    ("s = ", "''\n"),
    ("lst = ", "[]\n"),
    ("d = ", "{}\n"),
    ("total = ", "0\n"),
    ("flag = ", "True\n"),
    ("count = ", "0\n"),
    ("ans = ", "None\n"),
    
    # Category 4: Atomic Operations & Returns
    ("return ", "x\n"),
    ("return ", "0\n"),
    ("return ", "True\n"),
    ("return ", "False\n"),
    ("return ", "None\n"),
    ("return ", "a + b\n"),
    ("return ", "x * x\n"),
    ("total +", "= x\n"),
    ("i +", "= 1\n"),
    
    # Category 5: Primitive Control Skeletons
    ("if x:", "\n    return True\n"),
    ("if not x:", "\n    return None\n"),
    ("for x in arr:", "\n    pass\n"),
    ("while True:", "\n    break\n"),
    ("def f():", "\n    pass\n")
]

class PhonicsDataset(Dataset):
    def __init__(self, items: List[Tuple[str, str]], tokenizer: ByteCodeTokenizer, prefix_len: int = 16, context_len: int = 32):
        self.tokenizer = tokenizer
        self.prefix_len = prefix_len
        self.context_len = context_len
        self.samples = []
        pad_id = tokenizer.pad_token_id
        
        for prompt, target in items:
            p_ids = tokenizer.encode(prompt)
            p_ids = p_ids[-prefix_len:] if len(p_ids) > prefix_len else p_ids + [pad_id] * (prefix_len - len(p_ids))
            
            t_ids = tokenizer.encode(target)
            t_ids = t_ids[:context_len] if len(t_ids) > context_len else t_ids + [pad_id] * (context_len - len(t_ids))
            
            self.samples.append({
                "prefix": torch.tensor(p_ids, dtype=torch.long),
                "target": torch.tensor(t_ids, dtype=torch.long),
                "prompt_str": prompt,
                "target_str": target
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


class BabyNursery:
    """
    Nursery environment where the fresh AI baby learns its first characters,
    syllables, and Python phonics.
    """
    def __init__(self, device: Optional[torch.device] = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.tokenizer = ByteCodeTokenizer()
        
        # Lightweight Baby Config (compact scales for fast nursery learning)
        self.config = NCMConfig(
            context_length=32,
            prefix_length=16,
            levels=[2, 4, 16, 32],
            embed_dim=64,
            vq_codebook_size=128,
            vq_enc_layers=1,
            vq_dec_layers=1,
            gen_n_layer=2,
            gen_n_head=2,
            batch_size=16,
            lr_vqvae=2e-3,
            lr_gen=1.5e-3
        )
        
        # Initialize Baby Models
        self.vqvae = CodeVQVAE(
            vocab_size=self.tokenizer.vocab_size,
            context_length=self.config.context_length,
            embed_dim=self.config.embed_dim,
            levels=self.config.levels,
            codebook_size=self.config.vq_codebook_size,
            n_enc_layers=self.config.vq_enc_layers,
            n_dec_layers=self.config.vq_dec_layers
        ).to(device)
        
        self.generator = NextConceptGenerator(
            vqvae=self.vqvae,
            prefix_len=self.config.prefix_length,
            d_model=self.config.embed_dim,
            n_head=self.config.gen_n_head,
            n_layer=self.config.gen_n_layer,
            codebook_size=self.config.vq_codebook_size
        ).to(device)

    def test_baby_babble(self, test_prompts: List[str]) -> List[str]:
        """Observes what the baby babbles when prompted with letters or symbols."""
        self.generator.eval()
        outputs = []
        pad_id = self.tokenizer.pad_token_id
        P = self.config.prefix_length
        
        for prompt in test_prompts:
            raw_ids = self.tokenizer.encode(prompt)
            p_ids = raw_ids[-P:] if len(raw_ids) > P else raw_ids + [pad_id] * (P - len(raw_ids))
            prefix_t = torch.tensor([p_ids], dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                _, progression = self.generator.generate(
                    prefix_t,
                    temperature=0.2,
                    top_k=5,
                    top_p=0.9
                )
            final_tokens = progression[-1][0]
            clean_text = self.tokenizer.decode(final_tokens, skip_special_tokens=True)
            outputs.append(clean_text)
            
        return outputs

    def teach_phonics(self, epochs: int = 10, verbose: bool = True):
        """Trains the baby on the phonics primer until it masters syllables and symbols."""
        # Expand dataset
        expanded = []
        for _ in range(15):
            expanded.extend(PHONICS_PRIMER)
        random.seed(42)
        random.shuffle(expanded)
        
        dataset = PhonicsDataset(
            expanded,
            self.tokenizer,
            prefix_len=self.config.prefix_length,
            context_len=self.config.context_length
        )
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Step 1: Train Baby VQ-VAE on phonics
        if verbose:
            print("[Stage 1.1] Teaching the Baby Ear & Mouth (Character Reconstruction VQ-VAE)...", flush=True)
            
        opt_v = AdamW(self.vqvae.parameters(), lr=self.config.lr_vqvae)
        for ep in range(1, 7):
            self.vqvae.train()
            total_loss = 0.0
            for batch in loader:
                target = batch["target"].to(self.device)
                opt_v.zero_grad()
                logits, vq_loss, _ = self.vqvae(target)
                recon_loss = F.cross_entropy(logits.reshape(-1, self.tokenizer.vocab_size), target.reshape(-1))
                loss = recon_loss + vq_loss
                loss.backward()
                opt_v.step()
                total_loss += loss.item()
            if verbose:
                print(f"   Baby Phonics Ear Epoch {ep:02d}/06 | Loss: {total_loss/len(loader):.4f}", flush=True)
                
        self.vqvae.eval()
        for p in self.vqvae.parameters():
            p.requires_grad = False
            
        # Step 2: Train Baby Concept Generator
        if verbose:
            print("\n[Stage 1.2] Teaching the Baby Brain (Predicting Syllables in Parallel)...", flush=True)
            
        opt_g = AdamW(self.generator.parameters(), lr=self.config.lr_gen)
        for ep in range(1, epochs + 1):
            self.generator.train()
            total_loss = 0.0
            for batch in loader:
                prefix = batch["prefix"].to(self.device)
                target = batch["target"].to(self.device)
                opt_g.zero_grad()
                _, _, loss = self.generator(prefix, target)
                loss.backward()
                opt_g.step()
                total_loss += loss.item()
            if verbose:
                print(f"   Baby Speech Epoch {ep:02d}/{epochs:02d} | Loss: {total_loss/len(loader):.4f}", flush=True)
                
        # Save Baby Checkpoint
        os.makedirs("checkpoints/nursery", exist_ok=True)
        torch.save({
            "vqvae": self.vqvae.state_dict(),
            "generator": self.generator.state_dict(),
            "config": self.config
        }, "checkpoints/nursery/stage1_baby.pt")
        
        if verbose:
            print("\n[OK] Stage 1 Baby Checkpoint saved to checkpoints/nursery/stage1_baby.pt!", flush=True)
