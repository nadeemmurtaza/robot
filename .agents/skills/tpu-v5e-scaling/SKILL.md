---
name: tpu-v5e-scaling
description: >-
  Procedures for scaling, training, and deploying Code Robot on Google Cloud TPU v5e-1
  and Google Colab. Use when the user asks about TPU v5e-1 acceleration, bfloat16 training,
  scaling to 2.0B parameters, or operating with a 1 Million (1M) token context window.
---

# ⚡ TPU v5e-1 Scaling & 1M Context Skill

This skill provides step-by-step guidance for scaling **Code Robot** to **2.04 Billion parameters** with a **1 Million token (1,048,576) context window** on **Google Cloud TPU v5e-1** (16 GB HBM2) and Google Colab.

---

## 🖥️ Target Hardware: Google Cloud TPU v5e-1

- **TPU Chip**: TPU v5e (Single core: `v5e-1`)
- **High-Bandwidth Memory**: 16 GB HBM2
- **Native Precision**: `bfloat16` (native Tensor Processing Unit systolic arrays)
- **Framework**: PyTorch/XLA (`torch_xla`)
- **Memory Footprint for 2.04B Model**:
  $$\text{Memory} = 2,043,895,808 \times 2\text{ bytes} \approx 4.09\text{ GB}$$
  - Leaves **~11.9 GB of ultra-fast HBM2 free** for long-context KV caching and micro-batches.

---

## 🌌 1M Context Math & Zero-Memory RoPE

### Why Naive Embeddings Fail at 1M Tokens:
$$\text{Naive Table Memory} = 1,048,576 \times 2048 \times 4\text{ bytes} \approx 8.58\text{ GB}$$
A naive lookup table would consume more than half of the TPU's memory before computing a single attention operation!

### The RoPE Solution ($\theta = 10,000,000$):
Code Robot uses **Rotary Position Embeddings (RoPE)** with base frequency $\theta = 10,000,000$:
- **0 bytes lookup table**: Sines and cosines are computed on the fly on device.
- **Stable extrapolation**: Frequency base of $10^7$ prevents phase wrap-around at 1,048,576 tokens.
- **Dynamic Routing**: Automatically activated in `CodeVQVAE`, `NextConceptGenerator`, `LatentConsistencyPlanner`, and `HybridConsistencyNCM` whenever context length exceeds 8,192 tokens.

---

## 🚀 Execution Workflows

### 1. Verify 2.04B Parameters & 1M Context Locally
Verify the frontier architecture instantiation without allocating RAM using PyTorch's `meta` device:

```powershell
python robot.py --tiers
```

*Expected output highlights:*
- `Tier: Frontier Tier (2.04B Parameters / 1M Context)`
  - Total Parameters: `~2.07B` (VQ-VAE: 420.2M, Generator: 1.65B)
  - Context Window: `1,048,576` tokens (Prefix: `131,072` tokens)
  - Levels: `[64, 1024, 32768, 1048576]`
- `Tier: HC-NCM Frontier (2.0B+ / 1M Context)`:
  - Total Parameters: `2,043,895,808 (~2.04B)`
  - Position Scheme: `RoPE (theta = 10,000,000)`
  - TPU Memory: `~4.09 GB in bfloat16`

---

### 2. Google Colab TPU v5e-1 Workflow
1. Open the repository notebook in Google Colab:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nadeemmurtaza/robot/blob/main/robot_colab.ipynb)
2. Verify that the runtime is set to **TPU v5e**:
   - `Runtime` -> `Change runtime type` -> select `TPU v5e`.
3. In Step 1 of the notebook, verify device detection:
   ```python
   import torch_xla.core.xla_model as xm
   dev = xm.xla_device()
   print("TPU Active:", dev)
   ```
4. Run training or benchmark cells:
   ```bash
   !python run_hybrid_pipeline.py
   !python robot.py --tiers
   ```

---

### 3. Synchronizing Changes Between Colab and Local Workstation
When the user executes or saves in Colab:
```powershell
# Fetch any commits made from Colab
git fetch origin main

# Rebase local commits cleanly
git pull --rebase origin main

# Push updates to remote
git push origin main
```
