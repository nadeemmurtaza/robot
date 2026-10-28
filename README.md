# 🤖 Code Robot: Custom Next-Concept Modeling (NCM) for Code

A from-scratch, parallel coarse-to-fine hierarchical neural code generator inspired by Stanford Hazy Research ([arXiv:2610.14162](https://arxiv.org/abs/2610.14162)).

Unlike standard autoregressive transformers that generate tokens one by one ($T$ forward passes for $T$ tokens), **Next-Concept Modeling (NCM)** generates code hierarchically in coarse-to-fine levels with **bidirectional parallel attention within each level**.

---

## 🚀 Key Highlights & Parameter Tiers

Code Robot supports flexible model scaling from compact edge/CPU development up to deep high-capacity representations:

| Model Tier | Total Params | VQ-VAE Concept Tokenizer | NCM Generator Transformer | Target Context / Prefix | Hierarchy Levels |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Small** (Baseline) | **~2.9M** | 974K (2 enc / 2 dec) | 1.95M (4 layers, 4 heads, $d=128$) | 64 / 32 tokens | `[2, 8, 32, 64]` |
| **Medium** (Scaled) | **~19.3M** | 6.98M (4 enc / 4 dec) | 12.36M (6 layers, 8 heads, $d=256$) | 128 / 64 tokens | `[4, 16, 64, 128]` |
| **Large** (Deep) | **~108.3M** | 40.32M (6 enc / 6 dec) | 67.95M (8 layers, 8 heads, $d=512$) | 256 / 64 tokens | `[4, 16, 64, 256]` |

---

## 💡 How Next-Concept Generation Works

Code generation proceeds coarse-to-fine in only **4 forward evaluations**:

```
[Prompt Prefix: e.g. "def filter_evens(nums):"]
      │
      ▼ (Evaluation 0)
[Level 0: Global Concept]         Scale: 2 tokens    (Generated in parallel)
      │
      ▼ (Evaluation 1)
[Level 1: Structural Outline]      Scale: 8 tokens    (Generated in parallel)
      │
      ▼ (Evaluation 2)
[Level 2: Statements & Blocks]    Scale: 32 tokens   (Generated in parallel)
      │
      ▼ (Evaluation 3)
[Level 3: Full Code Tokens]       Scale: 64 tokens   (Generated in parallel)
```

- **Speedup**: A 64-token code body is generated in **4 forward passes** instead of 64 sequential passes (**16x fewer generation steps**).
- **Parallel Attention**: Within each level, attention is fully bidirectional, enabling global code awareness across the entire block.

---

## 📁 Repository Structure

```
robot/
├── custom_ncm/                 # Custom Next-Concept Modeling package
│   ├── config.py              # Configuration dataclasses & parameter tiers (Small/Medium/Large)
│   ├── code_tokenizer.py      # Byte-level UTF-8 code tokenizer
│   ├── dataset.py             # Python dataset generator & pairing
│   ├── vqvae.py               # Multiscale Residual Vector Quantizer (Stage 1)
│   ├── ncm_model.py           # Next-Concept Transformer Generator (Stage 2)
│   ├── self_trainer.py        # Autonomous self-training engine with AST & unit testing
│   ├── stage1_nursery.py      # Developmental Stage 1 phonics training
│   ├── teacher.py             # Interactive curriculum teacher engine
│   ├── curriculum.py          # 10-chapter progressive Python curriculum
│   ├── task_bank.py           # Programmatic coding tasks & test verification
│   ├── verifier.py            # AST syntax & execution sandbox
│   └── official_python_harvester.py # Harvests official Python standard library
├── checkpoints/               # Trained models and checkpoints
│   ├── generator_official_python.pt # Trained on official Python standard library
│   ├── generator.pt           # Next-concept generator
│   └── vqvae_pos.pt           # Position-aware concept VQ-VAE
├── robot.py                   # Main CLI entry point & model tier inspector
├── teach_robot.py             # Interactive teaching CLI (curriculum chapters 1-10)
├── run_stage1.py              # Stage 1 Nursery phonics runner
├── run_self_training.py       # Autonomous self-training runner
├── run_pipeline.py            # End-to-end training and inference demonstration
└── README.md
```

---

## 🛠️ Quickstart

### 1. View Model Parameter Tiers
```bash
python robot.py --tiers
```

### 2. Generate Code from a Prompt
```bash
python robot.py --prompt "# Filter even numbers\ndef filter_evens(nums):\n"
```

### 3. Interactive Teacher Mode (Teach Custom Functions)
```bash
python teach_robot.py --interactive
```

### 4. Teach Curriculum Chapters
```bash
python teach_robot.py --chapter 1      # Chapter 1: Arithmetic & Math
python teach_robot.py --all            # Complete 10-chapter syllabus
```

### 5. Launch Autonomous Self-Training
```bash
python run_self_training.py
```

---

## 📜 License
MIT License.
