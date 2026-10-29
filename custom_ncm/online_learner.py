"""
Online Learner: Real-time Live Training for Code Robot.
Allows Code Robot to ingest code snippets and documentation directly into its neural weights
as it browses and reads web pages.
"""
import os
import torch
from torch.optim import AdamW
from typing import List, Dict, Any, Optional

from custom_ncm.config import NCMConfig
from custom_ncm.code_tokenizer import ByteCodeTokenizer
from custom_ncm.vqvae import CodeVQVAE
from custom_ncm.ncm_model import NextConceptGenerator
from custom_ncm.dataset import CodeDataset
from torch.utils.data import DataLoader

class OnlineCodeLearner:
    """
    Live Online Learner for Code Robot.
    Takes scraped code blocks & text and immediately performs gradient backpropagation
    to train model weights while browsing.
    """
    def __init__(
        self,
        checkpoint_out: str = "checkpoints/live_trained_robot.pt",
        device: Optional[torch.device] = None,
        lr: float = 1e-4
    ):
        self.checkpoint_out = checkpoint_out
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = ByteCodeTokenizer()
        self.config = NCMConfig.small()
        self.lr = lr

        os.makedirs(os.path.dirname(self.checkpoint_out), exist_ok=True)

        # 1. Load VQ-VAE Concept Tokenizer
        vqvae_path = "checkpoints/vqvae_pos.pt" if os.path.exists("checkpoints/vqvae_pos.pt") else "checkpoints/vqvae.pt"
        self.vqvae = CodeVQVAE(
            vocab_size=self.tokenizer.vocab_size,
            context_length=self.config.context_length,
            embed_dim=self.config.embed_dim,
            levels=self.config.levels,
            codebook_size=self.config.vq_codebook_size,
            n_enc_layers=self.config.vq_enc_layers,
            n_dec_layers=self.config.vq_dec_layers
        ).to(self.device)

        if os.path.exists(vqvae_path):
            v_data = torch.load(vqvae_path, map_location=self.device, weights_only=False)
            self.vqvae.load_state_dict(v_data["model"])
        else:
            print(
                f"  [!] WARNING: no VQ-VAE checkpoint found at '{vqvae_path}'. "
                "Live training will run against an UNTRAINED, randomly-initialized concept "
                "tokenizer - learned codes will be meaningless. Run train_vqvae.py first."
            )
        self.vqvae.eval()
        for p in self.vqvae.parameters():
            p.requires_grad = False

        # 2. Load NCM Generator
        self.generator = NextConceptGenerator(
            vqvae=self.vqvae,
            prefix_len=self.config.prefix_length,
            d_model=self.config.embed_dim,
            n_head=self.config.gen_n_head,
            n_layer=self.config.gen_n_layer,
            codebook_size=self.config.vq_codebook_size
        ).to(self.device)

        # Load existing generator checkpoint if available
        base_ckpt = (
            self.checkpoint_out if os.path.exists(self.checkpoint_out)
            else ("checkpoints/generator_official_python.pt" if os.path.exists("checkpoints/generator_official_python.pt")
                  else "checkpoints/generator.pt")
        )
        if os.path.exists(base_ckpt):
            g_data = torch.load(base_ckpt, map_location=self.device, weights_only=False)
            self.generator.load_state_dict(g_data["model"], strict=False)
        else:
            print(
                f"  [!] WARNING: no generator checkpoint found (looked for '{self.checkpoint_out}', "
                "'checkpoints/generator_official_python.pt', 'checkpoints/generator.pt'). "
                "Starting live training from a randomly-initialized generator."
            )

        self.generator.train()
        self.optimizer = AdamW(self.generator.parameters(), lr=self.lr, weight_decay=0.01)

    def train_on_page(self, title: str, code_blocks: List[Dict[str, str]], text_paragraphs: List[str]) -> Dict[str, Any]:
        """
        Converts scraped code blocks and contextual paragraphs into prompt-target training pairs,
        runs backpropagation, and updates model weights in real time.
        """
        pairs = []
        context_len = self.config.context_length
        skipped_too_long = 0

        # 1. Code block pairs
        for b in code_blocks:
            code = b.get("code", "").strip()
            context = b.get("context", "").strip() or f"# Code from {title}"
            if len(code) > 10:
                # Target is truncated to context_len bytes by CodeDataset - a snippet much
                # longer than that would be cut mid-statement into a misleading training
                # target, so skip it instead of teaching the model a chopped-off fragment.
                if len(code.encode("utf-8")) > context_len:
                    skipped_too_long += 1
                    continue
                # Use context/comment as prompt, code as target
                prompt = context if context.startswith("#") else f"# {context}\n"
                pairs.append((prompt, code))

        if skipped_too_long:
            print(
                f"  [!] Skipped {skipped_too_long} code block(s) longer than "
                f"context_length={context_len} bytes (would be truncated mid-statement)."
            )

        # 2. Text paragraphs pairs (teaches syntax and natural language explanations)
        for p in text_paragraphs[:10]:
            if len(p) > 30 and ("def " in p or "=" in p or "import " in p):
                pairs.append((f"# Note from {title}\n", p))

        if not pairs:
            return {"loss": None, "trained_samples": 0}

        dataset = CodeDataset(
            pairs,
            self.tokenizer,
            prefix_len=self.config.prefix_length,
            context_len=self.config.context_length
        )
        loader = DataLoader(dataset, batch_size=min(len(pairs), 8), shuffle=True)

        total_loss = 0.0
        steps = 0

        self.generator.train()
        for batch in loader:
            prefix_ids = batch["prefix"].to(self.device)
            target_ids = batch["target"].to(self.device)

            self.optimizer.zero_grad()
            _, _, loss = self.generator(prefix_ids, target_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(steps, 1)

        # Save live weights
        torch.save({
            "model": self.generator.state_dict(),
            "config": self.config,
            "trained_samples": len(pairs)
        }, self.checkpoint_out)

        return {
            "loss": avg_loss,
            "trained_samples": len(pairs),
            "checkpoint": self.checkpoint_out
        }
