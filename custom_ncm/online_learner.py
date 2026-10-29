"""
Online Learner: Real-time Live Training for Code Robot.
Allows Code Robot to ingest code snippets and documentation directly into its neural weights
as it browses and reads web pages.
"""
import os
import json
import random
import hashlib
import torch
from torch.optim import AdamW
from typing import List, Dict, Any, Optional, Tuple

from custom_ncm.config import NCMConfig
from custom_ncm.code_tokenizer import ByteCodeTokenizer
from custom_ncm.vqvae import CodeVQVAE
from custom_ncm.ncm_model import NextConceptGenerator
from custom_ncm.dataset import CodeDataset
from custom_ncm.attention import convert_legacy_mha_state_dict
from custom_ncm.verifier import CodeVerifier
from custom_ncm.generate import generate_code_coarse_to_fine
from custom_ncm.train_vqvae import get_device, _warn_if_hardware_insufficient
from torch.utils.data import DataLoader

# Fixed, site-independent prompts used to measure whether live training is
# actually making the model better at writing real code, not just lowering
# loss on whatever page it last saw. Same prompts every call -> a comparable
# before/after number regardless of which website was scraped.
PROBE_PROMPTS = [
    "# Return the sum of all elements in a list\ndef total(nums):\n",
    "# Check if a number is even\ndef is_even(n):\n",
    "# Reverse a string\ndef reverse_str(s):\n",
    "# Return the length of the longest word in a list\ndef longest_word(words):\n",
    "# Remove all whitespace from a string\ndef strip_spaces(s):\n",
]

# Data-labeling outcomes that keep the candidate (everything else is rejected)
KEPT_LABELS = ("code", "code_truncated", "note")

class OnlineCodeLearner:
    """
    Live Online Learner for Code Robot.
    Takes scraped code blocks & text and immediately performs gradient backpropagation
    to train model weights while browsing.
    """
    def __init__(
        self,
        tier: str = "small",
        checkpoint_out: Optional[str] = None,
        device: Optional[torch.device] = None,
        lr: float = 1e-4,
        replay_path: Optional[str] = None,
        replay_max_size: int = 2000,
        replay_sample_size: int = 8
    ):
        self.tier = tier
        self.config = NCMConfig.from_tier(tier)
        self.device = device or get_device()
        self.tokenizer = ByteCodeTokenizer()
        self.lr = lr
        _warn_if_hardware_insufficient(self.config, self.device)

        # Checkpoints/replay buffer are tier-suffixed: a small-tier generator
        # has different shapes than the 2B/1M-context one, so sharing one
        # filename across tiers would load mismatched weights or crash on
        # shape mismatch (see train_on_official_python.py for the same fix).
        tier_suffix = "" if tier == "small" else f"_{tier}"
        self.checkpoint_out = checkpoint_out or f"checkpoints/live_trained_robot{tier_suffix}.pt"
        self.replay_path = replay_path or f"checkpoints/replay_buffer{tier_suffix}.jsonl"

        os.makedirs(os.path.dirname(self.checkpoint_out), exist_ok=True)

        # Replay buffer: each live training step on a new page is also mixed
        # with a random sample of PAST pages' (prompt, code) pairs, loaded
        # from/persisted to disk. Without this, repeated single-page gradient
        # updates with no held-out data drift the weights toward whatever was
        # browsed most recently and forget earlier pages (catastrophic
        # forgetting) - this is the standard experience-replay mitigation.
        self.replay_max_size = replay_max_size
        self.replay_sample_size = replay_sample_size
        self.replay_buffer: List[Tuple[str, str]] = self._load_replay_buffer()

        # Data-labeling step state: tracks code hashes already trained on (so
        # the same snippet copy-pasted across pages isn't labeled "fresh data"
        # every time) across the life of this learner.
        self.seen_code_hashes: set = set()

        # 1. Load VQ-VAE Concept Tokenizer (tier-suffixed, same reasoning as
        # the checkpoints above - a small-tier vqvae can't load into a
        # 2B/1M-context CodeVQVAE and vice versa)
        # NOTE: deliberately no fallback to the unsuffixed (small-tier)
        # checkpoint here - that checkpoint has different tensor shapes for
        # any OTHER tier and would crash load_state_dict with a confusing
        # size-mismatch error instead of just training fresh (bit us once
        # while testing the "web" tier: it silently tried to load
        # vqvae_pos.pt's 128-dim weights into a 256-dim web-tier model).
        default_vqvae = f"checkpoints/vqvae_pos{tier_suffix}.pt"
        if not os.path.exists(default_vqvae) and os.path.exists(f"checkpoints/vqvae{tier_suffix}.pt"):
            default_vqvae = f"checkpoints/vqvae{tier_suffix}.pt"
        vqvae_path = default_vqvae
        self.vqvae = CodeVQVAE(
            vocab_size=self.tokenizer.vocab_size,
            context_length=self.config.context_length,
            embed_dim=self.config.embed_dim,
            levels=self.config.levels,
            codebook_size=self.config.vq_codebook_size,
            n_enc_layers=self.config.vq_enc_layers,
            n_dec_layers=self.config.vq_dec_layers,
            attn_window=self.config.attn_window,
            attn_num_global=self.config.attn_num_global
        ).to(self.device)

        if os.path.exists(vqvae_path):
            v_data = torch.load(vqvae_path, map_location=self.device, weights_only=False)
            self.vqvae.load_state_dict(convert_legacy_mha_state_dict(v_data["model"]))
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
            codebook_size=self.config.vq_codebook_size,
            attn_window=self.config.attn_window,
            attn_num_global=self.config.attn_num_global
        ).to(self.device)

        # Load existing generator checkpoint if available (tier-suffixed)
        base_ckpt = (
            self.checkpoint_out if os.path.exists(self.checkpoint_out)
            else (f"checkpoints/generator_official_python{tier_suffix}.pt" if os.path.exists(f"checkpoints/generator_official_python{tier_suffix}.pt")
                  else f"checkpoints/generator{tier_suffix}.pt")
        )
        if os.path.exists(base_ckpt):
            g_data = torch.load(base_ckpt, map_location=self.device, weights_only=False)
            self.generator.load_state_dict(convert_legacy_mha_state_dict(g_data["model"]), strict=False)
        else:
            print(
                f"  [!] WARNING: no generator checkpoint found (looked for '{self.checkpoint_out}', "
                f"'checkpoints/generator_official_python{tier_suffix}.pt', "
                f"'checkpoints/generator{tier_suffix}.pt'). "
                "Starting live training from a randomly-initialized generator."
            )

        self.generator.train()
        self.optimizer = AdamW(self.generator.parameters(), lr=self.lr, weight_decay=0.01)

    def weight_snapshot(self) -> Dict[str, Any]:
        """
        Flat copy of every trainable generator weight plus its L2 norm. Call
        once before training; pass the result to weight_drift() afterward to
        measure how much the weights actually moved - a model whose loss
        looks like it's improving but whose weights barely move (or whose
        weights have exploded) is a real failure mode live training can hit.
        """
        with torch.no_grad():
            flat = torch.cat([p.detach().flatten() for p in self.generator.parameters()]).clone()
        return {"l2_norm": flat.norm().item(), "_flat": flat}

    def weight_drift(self, snapshot_before: Dict[str, Any]) -> Dict[str, Any]:
        with torch.no_grad():
            flat_now = torch.cat([p.detach().flatten() for p in self.generator.parameters()])
            before = snapshot_before["_flat"].to(flat_now.device)
            l2_delta = (flat_now - before).norm().item()
            cos_sim = torch.nn.functional.cosine_similarity(
                flat_now.unsqueeze(0), before.unsqueeze(0)
            ).item()
        return {
            "l2_norm_before": snapshot_before["l2_norm"],
            "l2_norm_after": flat_now.norm().item(),
            "l2_delta": l2_delta,
            "cosine_similarity_to_before": cos_sim,
        }

    def _load_replay_buffer(self) -> List[Tuple[str, str]]:
        if not os.path.exists(self.replay_path):
            return []
        buffer = []
        with open(self.replay_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    buffer.append((item["prompt"], item["target"]))
        # Keep only the most recent replay_max_size on load too, in case the
        # file was grown by an older run with a larger cap.
        return buffer[-self.replay_max_size:]

    def _add_to_replay_buffer(self, pairs: List[Tuple[str, str]]) -> None:
        if not pairs:
            return
        self.replay_buffer.extend(pairs)
        if len(self.replay_buffer) > self.replay_max_size:
            self.replay_buffer = self.replay_buffer[-self.replay_max_size:]
        with open(self.replay_path, "w", encoding="utf-8") as f:
            for prompt, target in self.replay_buffer:
                f.write(json.dumps({"prompt": prompt, "target": target}) + "\n")

    def evaluate(self, prompts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Real "is the model better" check: generate code for a fixed set of
        probe prompts and score it with CodeVerifier (syntax-valid %), plus
        the average per-token-position output length (a trivially-collapsed
        model - e.g. always predicting <pad> or an empty body - shows up here
        as near-zero length even when syntax happens to be valid). Call this
        before and after a training run and diff the two results.
        """
        prompts = prompts or PROBE_PROMPTS
        was_training = self.generator.training
        self.generator.eval()

        syntax_valid = 0
        total_len = 0
        for prompt in prompts:
            decoded_levels = generate_code_coarse_to_fine(
                generator=self.generator, tokenizer=self.tokenizer, prompt=prompt,
                prefix_len=self.config.prefix_length, temperature=0.0, verbose=False
            )
            gen_body = decoded_levels[-1]
            ok, _, _ = CodeVerifier.check_syntax(prompt + gen_body)
            syntax_valid += int(ok)
            total_len += len(gen_body.strip())

        if was_training:
            self.generator.train()

        n = max(len(prompts), 1)
        return {
            "n_probes": len(prompts),
            "syntax_valid_pct": 100.0 * syntax_valid / n,
            "avg_output_len": total_len / n,
        }

    def _fit_code_to_context(self, code: str, context_len: int) -> Tuple[str, bool]:
        """
        Every code block gets accepted regardless of length - CodeDataset
        would otherwise blindly byte-slice to context_len mid-statement, so
        this tries to land on a whole-line prefix that still parses first
        (shrinking line by line from the end), and only falls back to a hard
        byte truncation if no whole-line prefix parses on its own. Returns
        (fitted_code, was_truncated).
        """
        encoded = code.encode("utf-8")
        if len(encoded) <= context_len:
            return code, False

        lines = code.splitlines(keepends=True)
        prefix = ""
        for line in lines:
            candidate = prefix + line
            if len(candidate.encode("utf-8")) > context_len:
                break
            prefix = candidate

        candidate_lines = prefix.splitlines(keepends=True)
        while candidate_lines:
            candidate = "".join(candidate_lines)
            ok, _, _ = CodeVerifier.check_syntax(candidate)
            if ok:
                return candidate, True
            candidate_lines.pop()

        # Nothing parses as a standalone whole-line prefix - accept a hard
        # truncation anyway rather than reject the example outright.
        return encoded[:context_len].decode("utf-8", errors="ignore"), True

    def label_scraped_data(
        self, title: str, code_blocks: List[Dict[str, str]], text_paragraphs: List[str]
    ) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
        """
        Data-labeling step: turns raw scraped (code_block, paragraph) material
        into labeled (prompt, target) training pairs. Every code block is
        accepted regardless of length (see _fit_code_to_context) - only
        genuinely bad data gets rejected. Every candidate gets exactly one
        label:
          - "code"             : kept - syntactically valid Python, as-is
          - "code_truncated"    : kept - too long for context_length, fit to
                                  it (preferring a still-valid-syntax prefix)
          - "code_bad_syntax"  : rejected - ast.parse fails even after trying
                                  to truncate to a valid prefix (scraped
                                  garbage, pseudo-code, shell output
                                  mis-tagged as python, etc.)
          - "code_too_short"   : rejected - under 10 chars, not a real example
          - "code_duplicate"   : rejected - exact (possibly truncated) snippet
                                  already labeled on an earlier page this run
          - "note"             : kept - prose paragraph that reads like it
                                  documents code (has def/=/import in it)
        Returns (kept_pairs, label_counts) - label_counts is reported back up
        so the caller can print/aggregate a labeling report.
        """
        context_len = self.config.context_length
        kept_pairs: List[Tuple[str, str]] = []
        label_counts: Dict[str, int] = {}

        def label(name: str):
            label_counts[name] = label_counts.get(name, 0) + 1

        for b in code_blocks:
            code = b.get("code", "").strip()
            context = b.get("context", "").strip() or f"# Code from {title}"

            if len(code) <= 10:
                label("code_too_short")
                continue

            code, was_truncated = self._fit_code_to_context(code, context_len)

            code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
            if code_hash in self.seen_code_hashes:
                label("code_duplicate")
                continue

            ok, _, _ = CodeVerifier.check_syntax(code)
            if not ok:
                label("code_bad_syntax")
                continue

            self.seen_code_hashes.add(code_hash)
            label("code_truncated" if was_truncated else "code")
            prompt = context if context.startswith("#") else f"# {context}\n"
            kept_pairs.append((prompt, code))

        # Text paragraphs that read like they document code (no syntax check -
        # these are natural-language targets, not Python).
        for p in text_paragraphs[:10]:
            if len(p) > 30 and ("def " in p or "=" in p or "import " in p):
                label("note")
                kept_pairs.append((f"# Note from {title}\n", p))

        return kept_pairs, label_counts

    def train_on_page(self, title: str, code_blocks: List[Dict[str, str]], text_paragraphs: List[str]) -> Dict[str, Any]:
        """
        Labels scraped code blocks and contextual paragraphs into prompt-target
        training pairs (see label_scraped_data), then runs backpropagation and
        updates model weights in real time on the pairs that pass labeling.
        """
        pairs, label_counts = self.label_scraped_data(title, code_blocks, text_paragraphs)

        rejected = sum(v for k, v in label_counts.items() if k not in KEPT_LABELS)
        if rejected:
            reasons = ", ".join(f"{k}={v}" for k, v in label_counts.items() if k not in KEPT_LABELS)
            print(f"  [*] Data labeling: rejected {rejected} candidate(s) ({reasons})")
        kept = sum(label_counts.get(k, 0) for k in KEPT_LABELS)
        if kept:
            print(
                f"  [*] Data labeling: kept {kept} labeled pair(s) "
                f"(code={label_counts.get('code', 0)}, "
                f"truncated={label_counts.get('code_truncated', 0)}, "
                f"note={label_counts.get('note', 0)})"
            )

        if not pairs:
            return {"loss": None, "trained_samples": 0, "label_counts": label_counts}

        new_pairs = list(pairs)

        # Mix in a random sample of past pages' pairs so this gradient step
        # doesn't only see the current page (replay buffer - see __init__).
        replay_sample = []
        if self.replay_buffer:
            k = min(self.replay_sample_size, len(self.replay_buffer))
            replay_sample = random.sample(self.replay_buffer, k)
            pairs = pairs + replay_sample

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

        # Add this page's (non-replay) pairs to the buffer for future pages.
        self._add_to_replay_buffer(new_pairs)

        # Save live weights
        torch.save({
            "model": self.generator.state_dict(),
            "config": self.config,
            "trained_samples": len(pairs)
        }, self.checkpoint_out)

        return {
            "loss": avg_loss,
            "trained_samples": len(pairs),
            "replay_samples_mixed_in": len(replay_sample),
            "replay_buffer_size": len(self.replay_buffer),
            "label_counts": label_counts,
            "checkpoint": self.checkpoint_out
        }
