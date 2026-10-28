---
name: train-code-robot
description: >-
  Procedures for training, fine-tuning, and evaluating Code Robot across all architectural tiers.
  Use when the user asks to train Code Robot, train on official Python documentation/stdlib,
  run the autonomous self-training reinforcement loop, or train the Stage-1 VQ-VAE and Stage-2 NCM generator.
---

# 🏋️ Train Code Robot Skill

This skill provides complete runbooks for training, fine-tuning, and evaluating **Code Robot** across all stages and parameter tiers.

---

## 🏗️ The 2-Stage Training Architecture

```
[Raw Python Code]
       │
       ▼
[Stage 1: Concept VQ-VAE] ──> Discretizes code into hierarchical concept codes [c_0, c_1, c_2, c_3]
       │                       Loss: Reconstruction Cross-Entropy + Commitment Loss
       ▼
[Stage 2: NCM Generator]  ──> Predicts concept pyramids conditioned on prompt prefix
                               Loss: Multi-scale Hierarchical Cross-Entropy Loss
```

---

## 📋 Training Runbooks

### 1. Train on Official Python Standard Library & Docs
Trains on the 1,415 verified functions harvested across 419 official Python standard library modules:

```powershell
python train_on_official_python.py
```

*Key Parameters:*
- Default epochs: `6`
- Dataset: `data/official_python_corpus.json`
- Target checkpoint: `checkpoints/generator_official_python.pt`
- Expected loss progression: Drops from `~9.4` down to `~7.1` or lower.

---

### 2. Autonomous Self-Training Loop (Compiler-in-the-Loop Reinforcement)
Runs self-directed code generation, syntax validation, and reinforcement learning:

```powershell
python run_self_training.py
```

*How it works:*
1. Generates code candidates across varying temperature and top-k settings.
2. Runs programmatic `ast.parse()` and sandbox execution tests.
3. Automatically saves valid, passing implementations back into the training corpus.
4. Updates model weights based on successful executions.

---

### 3. Stage 1: Multiscale Concept VQ-VAE Training
Trains the concept tokenizer that maps token sequences to hierarchical discrete latents:

```powershell
python -c "
import torch
from custom_ncm.config import NCMConfig
from custom_ncm.vqvae import CodeVQVAE
from custom_ncm.device import get_optimal_device

dev, dtype = get_optimal_device()
cfg = NCMConfig.small()
model = CodeVQVAE(
    vocab_size=cfg.vocab_size,
    embed_dim=cfg.embed_dim,
    levels=cfg.levels,
    context_length=cfg.context_length,
    codebook_size=cfg.vq_codebook_size
).to(dev)
print(f'Training VQ-VAE on {dev} with {sum(p.numel() for p in model.parameters()):,} parameters...')
"
```

---

### 4. Stage 2: Next-Concept Modeling (NCM) Generator Training
Trains the hierarchical generator with block-causal attention masks:

```powershell
python -c "
import torch
from custom_ncm.config import NCMConfig
from custom_ncm.ncm_model import NextConceptGenerator
from custom_ncm.vqvae import CodeVQVAE
from custom_ncm.device import get_optimal_device

dev, dtype = get_optimal_device()
cfg = NCMConfig.small()
vqvae = CodeVQVAE(256, cfg.context_length, cfg.embed_dim, cfg.levels, cfg.vq_codebook_size).to(dev)
gen = NextConceptGenerator(vqvae, prefix_len=cfg.prefix_length, d_model=cfg.embed_dim).to(dev)
print(f'Training Generator on {dev} with {sum(p.numel() for p in gen.parameters()):,} parameters...')
"
```

---

## 📊 Verification & Diagnostics

After training, check generation quality:

```powershell
# Test on a standard function prompt
python robot.py --prompt "# Compute factorial of n\ndef factorial(n):\n"

# Check parameter tiers and active configuration
python robot.py --tiers
```
