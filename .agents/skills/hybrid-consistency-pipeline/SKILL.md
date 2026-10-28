---
name: hybrid-consistency-pipeline
description: >-
  Run, train, and benchmark the Hybrid Consistency Next-Concept Model (HC-NCM).
  Use when the user asks about 1-step Latent Consistency Mapping (LCM), bidirectional diffusion backbones,
  Diffusion Gemma architecture, Causal AR realization, or compiler-in-the-loop AST verification.
---

# ⚡ Hybrid Consistency Pipeline (HC-NCM) Skill

This skill guides the operation, training, and benchmarking of the **Hybrid Consistency Next-Concept Model (HC-NCM)** for ultrafast sub-120ms code generation.

---

## 🧩 Architectural Synergy

```
[Prompt Prefix]
      │
      ├──> [1-Step LCM Planner] (Diffusion Gemma Bidirectional AdaLN Backbone)
      │         │ Maps noise -> Clean Global Concept Blueprint in 1 forward evaluation
      │         ▼
      ├──> [NCM Pyramid] (Coarse-to-Fine Hierarchical Expansion across scales)
      │         ▼
      └──> [Causal AR Head] (Cross-Attends to Blueprint for exact Python syntax)
                ▼
           [Compiler Loop] (AST Parsing + Sandboxed Execution Validation)
```

---

## 🚀 Execution & Benchmarking Runbooks

### 1. Run the Full Architectural Benchmark
Benchmarks generation speed, forward evaluation count, and syntactic quality:

```powershell
python run_hybrid_pipeline.py
```

*Expected Output Metrics:*
- Forward passes for 64 tokens: **1 (LCM) + 4 (NCM) steps** (vs. 64 for standard autoregressive LLMs).
- Latency: **~110ms - 130ms**.
- Generates 1-step concept blueprint shape: `[1, 8, embed_dim]`.
- Prints multi-scale progression across Level 0, Level 1, Level 2, and Level 3.
- Runs programmatic AST validation.

---

### 2. Generate Code with the Hybrid Engine
Generate code for any custom prompt:

```powershell
python robot.py --hybrid --prompt "# Return the square of x`ndef square(x):`n"
```

*Diagnosing Output Sources:*
- `Chosen Source: Causal AR Head (Consistency-Guided)`: Causal decoder successfully realized syntax matching the consistency blueprint.
- `Chosen Source: NCM Parallel Fine Level`: Parallel fine level passed AST syntax checks directly.
- `AST Syntax Valid`: Indicates whether `ast.parse()` succeeded without errors.

---

### 3. Train the Hybrid Model Jointly
Trains the consistency loss, NCM concept cross-entropy, and causal token prediction simultaneously:

```powershell
python -c "
from custom_ncm.config import NCMConfig
from custom_ncm.train_hybrid import train_hybrid_model
from custom_ncm.device import get_optimal_device

dev, dtype = get_optimal_device()
cfg = NCMConfig.small()
print(f'Training HC-NCM on {dev}...')
train_hybrid_model(config=cfg, epochs=3, device=dev)
"
```

*Joint Loss Formula:*
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NCM concept}} + 0.5 \cdot \mathcal{L}_{\text{LCM consistency}} + 0.5 \cdot \mathcal{L}_{\text{Causal AR}}$$
