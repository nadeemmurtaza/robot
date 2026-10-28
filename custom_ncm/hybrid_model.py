"""
Hybrid Consistency Next-Concept Model (HC-NCM).
Unifies:
1. NCM: Multi-scale hierarchical concept pyramid (2 -> 8 -> 32 -> 64)
2. LCM: 1-step / 2-step direct trajectory mapping without slow diffusion loops
3. Diffusion Gemma: Full bidirectional look-ahead/look-behind attention for global planning
4. Causal Gemma/LLM Head: Autoregressive token realizer with cross-attention to concepts
5. AST Verifier: Programmatic syntax validation
"""
from typing import List, Tuple, Dict, Any, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_ncm.config import NCMConfig
from custom_ncm.code_tokenizer import ByteCodeTokenizer
from custom_ncm.vqvae import CodeVQVAE
from custom_ncm.ncm_model import NextConceptGenerator
from custom_ncm.lcm_flow import LatentConsistencyPlanner
from custom_ncm.verifier import CodeVerifier, VerificationResult
from custom_ncm.rope import RotaryPositionalEmbedding

class CausalTransformerBlock(nn.Module):
    """
    Causal Transformer Block with Cross-Attention (Gemma-style AR Decoder).
    Self-attention is strictly causal (lower-triangular).
    Cross-attention attends to the Consistency Concept blueprint.
    """
    def __init__(self, d_model: int, n_head: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        
        self.ln_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Causal Self-Attention
        normed = self.ln1(x)
        sa_out, _ = self.self_attn(normed, normed, normed, attn_mask=causal_mask)
        x = x + sa_out
        
        # Cross-Attention to LCM/NCM concept representations
        normed_cross = self.ln_cross(x)
        ca_out, _ = self.cross_attn(normed_cross, context, context)
        x = x + ca_out
        
        # Feedforward
        x = x + self.mlp(self.ln2(x))
        return x


class HybridConsistencyNCM(nn.Module):
    """
    The Full Hybrid Architecture:
    Combines LCM 1-Step Consistency + Diffusion Gemma Bidirectional Planner +
    NCM Hierarchical Pyramid + Causal LLM Decoder.
    """
    def __init__(
        self,
        config: NCMConfig,
        vqvae: CodeVQVAE,
        ncm_generator: Optional[NextConceptGenerator] = None
    ):
        super().__init__()
        self.config = config
        self.vqvae = vqvae
        self.embed_dim = config.embed_dim
        self.vocab_size = vqvae.vocab_size
        self.levels = config.levels
        self.prefix_len = config.prefix_length
        self.context_len = config.context_length
        
        # 1. Base NCM Generator for coarse-to-fine expansion
        if ncm_generator is not None:
            self.ncm_generator = ncm_generator
        else:
            self.ncm_generator = NextConceptGenerator(
                vqvae=vqvae,
                prefix_len=config.prefix_length,
                d_model=config.embed_dim,
                n_head=config.gen_n_head,
                n_layer=config.gen_n_layer,
                codebook_size=config.vq_codebook_size
            )
            
        # 2. Latent Consistency Planner (LCM + Diffusion Gemma bidirectional backbone)
        # Operates at Level 1 scale (e.g. 8 concept tokens)
        self.lcm_seq_len = self.levels[1] if len(self.levels) > 1 else 8
        self.lcm_planner = LatentConsistencyPlanner(
            latent_dim=self.embed_dim,
            concept_seq_len=self.lcm_seq_len,
            prefix_len=self.prefix_len,
            n_layers=3,
            n_heads=config.gen_n_head
        )
        
        # 3. Prompt Prefix Embedding
        self.prefix_token_emb = nn.Embedding(self.vocab_size, self.embed_dim)
        if self.prefix_len > 8192:
            self.prefix_rope = RotaryPositionalEmbedding(self.embed_dim, max_seq_len=self.prefix_len)
            self.prefix_pos_emb = None
        else:
            self.prefix_pos_emb = nn.Embedding(self.prefix_len, self.embed_dim)
            self.prefix_rope = None
        
        # 4. Causal Gemma-Style Token Decoder (Exact Syntax Realizer)
        self.target_token_emb = nn.Embedding(self.vocab_size, self.embed_dim)
        if self.context_len > 8192:
            self.target_rope = RotaryPositionalEmbedding(self.embed_dim, max_seq_len=self.context_len)
            self.target_pos_emb = None
        else:
            self.target_pos_emb = nn.Embedding(self.context_len, self.embed_dim)
            self.target_rope = None
            
        self.causal_blocks = nn.ModuleList([
            CausalTransformerBlock(self.embed_dim, n_head=config.gen_n_head)
            for _ in range(3)
        ])
        self.causal_ln = nn.LayerNorm(self.embed_dim)
        self.token_head = nn.Linear(self.embed_dim, self.vocab_size, bias=False)
        self.token_head.weight = self.target_token_emb.weight  # Weight tying

    def get_prefix_pos(self, x: torch.Tensor, P: int) -> torch.Tensor:
        if self.prefix_rope is not None:
            return self.prefix_rope(x, P)
        pos = torch.arange(P, device=x.device).unsqueeze(0)
        return self.prefix_pos_emb(pos)

    def get_target_pos(self, x: torch.Tensor, L: int) -> torch.Tensor:
        if self.target_rope is not None:
            return self.target_rope(x, L)
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        return self.target_pos_emb(pos)

    def encode_prefix(self, prefix_ids: torch.Tensor) -> torch.Tensor:
        """Embeds prompt prefix tokens with positional information."""
        B, P = prefix_ids.shape
        return self.prefix_token_emb(prefix_ids) + self.get_prefix_pos(prefix_ids, P)

    def forward(
        self,
        prefix: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        End-to-End Hybrid Training Forward Pass:
        1. VQ-VAE extracts target discrete concepts & continuous latents
        2. NCM Generator computes cross-entropy loss over concept hierarchy
        3. LCM Planner computes consistency loss on Level 1 concept trajectory
        4. Causal AR Head computes token prediction loss conditioned on concepts
        """
        device = prefix.device
        B, P = prefix.shape
        _, L = target.shape
        
        # 1. Target concepts from VQ-VAE
        with torch.no_grad():
            h_tgt = self.vqvae.token_emb(target) + self.vqvae.get_pos_emb(target, L)
            _, _, target_codes, z_intermediates = self.vqvae.vq(self.vqvae.enc_ln(h_tgt))
            
        # 2. Base NCM Generator Loss
        _, _, ncm_loss = self.ncm_generator(prefix, target)
        
        # 3. LCM Consistency Loss
        # Target representation at Level 1
        prefix_embeds = self.encode_prefix(prefix)
        # Downsample clean continuous latent to Level 1 scale
        clean_z0 = self.vqvae.vq.downsample(z_intermediates[1], self.lcm_seq_len)
        
        # Sample random timesteps sigma for consistency training
        sigmas = torch.rand(B, device=device) * 10.0 + 0.1
        noise = torch.randn_like(clean_z0)
        z_t = clean_z0 + noise * sigmas.view(-1, 1, 1)
        
        # Predict clean z_0 directly using 1-step consistency
        z_0_pred = self.lcm_planner(z_t, sigmas, prefix_embeds)
        lcm_loss = F.mse_loss(z_0_pred, clean_z0)
        
        # 4. Causal LLM Token Realizer Loss
        # Context consists of prefix embeddings concatenated with predicted concept latents
        concept_context = torch.cat([prefix_embeds, z_0_pred], dim=1)  # [B, P + S_lcm, D]
        
        # Autoregressive target input shifted by 1
        target_in = self.target_token_emb(target) + self.get_target_pos(target, L)
        
        # Causal mask (prevent seeing future tokens)
        causal_mask = torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)
        
        h = target_in
        for block in self.causal_blocks:
            h = block(h, concept_context, causal_mask=causal_mask)
        h = self.causal_ln(h)
        token_logits = self.token_head(h)
        
        token_loss = F.cross_entropy(token_logits.view(-1, self.vocab_size), target.view(-1))
        
        total_loss = ncm_loss + 0.5 * lcm_loss + 0.5 * token_loss
        
        return {
            "total_loss": total_loss,
            "ncm_loss": ncm_loss,
            "lcm_loss": lcm_loss,
            "token_loss": token_loss
        }

    @torch.no_grad()
    def fast_generate(
        self,
        prefix: torch.Tensor,
        tokenizer: ByteCodeTokenizer,
        mode: str = "hybrid",
        temperature: float = 0.3,
        top_k: int = 20
    ) -> Dict[str, Any]:
        """
        FAST HYBRID GENERATION:
        1. 1-Step Latent Consistency Mapping (LCM) to generate global concept blueprint
        2. Parallel Hierarchical Refinement (NCM) across scales
        3. Causal LLM Decoder Realization for exact syntax
        4. AST Syntax & Grammar Verification
        """
        self.eval()
        device = prefix.device
        B = prefix.shape[0]
        P = prefix.shape[1]
        
        prefix_embeds = self.encode_prefix(prefix)
        
        # Step 1: 1-Step LCM Global Blueprint Prediction
        # Uses bidirectional attention across the whole concept sequence
        lcm_concept_blueprint = self.lcm_planner.sample_one_step(prefix_embeds)
        
        # Step 2: NCM Hierarchical Progression
        _, ncm_progression = self.ncm_generator.generate(
            prefix,
            temperature=temperature,
            top_k=top_k
        )
        ncm_decoded_levels = [
            tokenizer.decode(level_tokens[0], skip_special_tokens=True)
            for level_tokens in ncm_progression
        ]
        
        # Step 3: Causal Gemma Token Realization (Conditioned on Consistency Blueprint)
        # Generates exact code tokens causally
        concept_context = torch.cat([prefix_embeds, lcm_concept_blueprint], dim=1)
        generated_tokens = []
        pad_id = tokenizer.pad_token_id
        
        # Start with empty target input and generate autoregressively with guidance
        curr_tokens = [pad_id]
        for step in range(self.context_len):
            curr_tensor = torch.tensor([curr_tokens], dtype=torch.long, device=device)
            seq_len = curr_tensor.shape[1]
            x_emb = self.target_token_emb(curr_tensor) + self.get_target_pos(curr_tensor, seq_len)
            
            causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
            h = x_emb
            for block in self.causal_blocks:
                h = block(h, concept_context, causal_mask=causal_mask)
            h = self.causal_ln(h)
            next_logits = self.token_head(h[:, -1, :]) / max(temperature, 1e-4)
            
            # Top-k sampling
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            curr_tokens.append(next_token)
            if next_token == pad_id or (next_token == ord('\n') and step > 40):
                # Stop if padding reached
                break
                
        causal_code = tokenizer.decode(curr_tokens[1:], skip_special_tokens=True)
        
        # Step 4: Pick best candidate (or blend NCM fine level with Causal Head)
        ncm_fine_code = ncm_decoded_levels[-1]
        
        # AST Validation check
        prompt_str = tokenizer.decode(prefix[0].tolist(), skip_special_tokens=True)
        v_ncm = CodeVerifier.run_tests(prompt_str + ncm_fine_code, "test_func", [])
        v_causal = CodeVerifier.run_tests(prompt_str + causal_code, "test_func", [])
        
        # Select best valid syntax
        if v_causal.is_valid_syntax:
            selected_body = causal_code
            chosen_source = "Causal AR Head (Consistency-Guided)"
        elif v_ncm.is_valid_syntax:
            selected_body = ncm_fine_code
            chosen_source = "NCM Parallel Fine Level"
        else:
            selected_body = ncm_fine_code
            chosen_source = "NCM Hierarchy (Fallback)"

        clean_prompt = prompt_str.strip()
        if not clean_prompt.startswith("def "):
            def_idx = prompt_str.find("def ")
            if def_idx != -1:
                clean_prompt = prompt_str[def_idx:]
                
        full_code = prompt_str + "\n" + selected_body if not prompt_str.endswith("\n") else prompt_str + selected_body
        final_ast = CodeVerifier.run_tests(full_code, "func", [])
        
        return {
            "prompt": prompt_str,
            "blueprint_lcm": lcm_concept_blueprint,
            "ncm_progression": ncm_decoded_levels,
            "causal_code": causal_code,
            "selected_body": selected_body,
            "chosen_source": chosen_source,
            "full_code": full_code,
            "is_valid_ast": final_ast.is_valid_syntax,
            "ast_error": final_ast.error_message
        }
