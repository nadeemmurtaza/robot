# Custom Next-Concept Modeling (NCM) for Code

A from-scratch, lightweight implementation of **Next-Concept Modeling (NCM)** designed specifically for code generation, inspired by Stanford Hazy Research ([arXiv:2610.14162](https://arxiv.org/abs/2610.14162)).

---

## 💡 The Core NCM Concept

Traditional Large Language Models (LLMs) generate code **autoregressively token-by-token**, requiring $T$ forward evaluations for a sequence of length $T$ (e.g. 64 forward passes for 64 tokens). Diffusion Language Models (DLMs) require evaluating the full sequence length at every denoising step.

**Next-Concept Modeling (NCM)** solves this through a **two-stage, coarse-to-fine hierarchical framework**:

```
[Prompt Prefix]
      │
      ▼ (Step 0)
[Level 0: Global Concept]         Scale: 2 tokens    (Generated in parallel)
      │
      ▼ (Step 1)
[Level 1: Structural Outline]      Scale: 8 tokens    (Generated in parallel)
      │
      ▼ (Step 2)
[Level 2: Statements & Blocks]    Scale: 32 tokens   (Generated in parallel)
      │
      ▼ (Step 3)
[Level 3: Full Code Tokens]       Scale: 64 tokens   (Generated in parallel)
```

- **Step Reduction**: A 64-token code snippet is generated in **only 4 forward passes** instead of 64 steps (**16x fewer generation steps**).
- **Parallelism**: Within each level, all discrete concept codes are generated simultaneously using bidirectional attention within the block and cross-attention back to the prompt and previous coarse levels.

---

## 📁 Repository Structure

```
my_code_robot/
├── custom_ncm/
│   ├── __init__.py
│   ├── config.py           # Dataclass configs (context lengths, levels, dims)
│   ├── code_tokenizer.py   # Byte-level tokenizer (UTF-8, 260 vocab) & HF support
│   ├── dataset.py          # Python code dataset generator with prompt/body pairs
│   ├── vqvae.py            # Multiscale Residual Vector Quantizer (Stage 1)
│   ├── ncm_model.py        # Next-Concept Transformer Generator (Stage 2)
│   ├── train_vqvae.py      # Stage 1 training script
│   ├── train_ncm.py        # Stage 2 training script
│   └── generate.py         # Coarse-to-fine generation & refinement visualizer
├── checkpoints/
│   ├── vqvae.pt            # Trained VQ-VAE model
│   └── generator.pt        # Trained NCM generator model
├── run_pipeline.py         # End-to-end training & evaluation pipeline
└── README_CUSTOM_NCM.md
```

---

## 🚀 Quickstart & Usage

### 1. Run Generation on Trained Model
To generate code from a prompt prefix and watch the coarse-to-fine refinement:

```bash
python -m custom_ncm.generate --prompt "# Filter even numbers from list\ndef filter_evens(nums):\n"
```

Or run interactively:
```bash
python -m custom_ncm.generate --interactive
```

### 2. Train Stage 1: Multiscale Concept VQ-VAE
Trains the code quantizer to encode code into discrete hierarchical levels `[2, 8, 32, 64]`:

```bash
python -m custom_ncm.train_vqvae
```

### 3. Train Stage 2: Next-Concept Transformer Generator
Trains the generator to predict concept codes given the prompt prefix and prior coarse levels:

```bash
python -m custom_ncm.train_ncm
```

### 4. Run the Full End-to-End Pipeline
```bash
python run_pipeline.py
```

---

## ⚙️ How It Works Mathematically

1. **Multiscale Residual VQ-VAE (Concept Tokenizer)**:
   - For an input sequence $x \in \mathbb{R}^{B \times L}$, encoder produces $z_e \in \mathbb{R}^{B \times L \times d}$.
   - Levels $S = [S_0, S_1, \dots, S_{K-1}]$ downsample residuals using area pooling and quantize against codebooks $E_l$.
   - Straight-through gradient estimator allows backprop: $\hat{z}_{STE} = z_e + (\hat{z} - z_e).\text{detach}()$.
   - VQ commitment loss minimizes $\| \text{sg}(\bar{r}_l) - e_l \|^2 + \beta \| \bar{r}_l - \text{sg}(e_l) \|^2$.

2. **Next-Concept Transformer (Generator)**:
   - Input layout: $[\text{Prefix } (P), \text{Level 0 Query } (S_0), Z^{(in)}_1 (S_1), \dots, Z^{(in)}_{K-1} (S_{K-1})]$.
   - Attention Mask:
     - Level $l$ attends to Prefix and all prior levels $< l$.
     - Within Level $l$, attention is **bidirectional**.
     - Future levels $> l$ are masked out ($-\infty$).
   - Generation:
     - Iterates through levels $l = 0 \dots K-1$.
     - In each step $l$, predicts all $S_l$ concept codes simultaneously in parallel.
     - Final tokens are decoded through the VQ-VAE decoder.
