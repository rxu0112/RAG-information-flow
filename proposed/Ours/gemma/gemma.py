# gemma.py
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

DATASET_SAMPLE_LIMITS = {
    "squad2": 50000,
    "msmarco": 45000,
    "hotpot": 40000,
}


def limit_dataset_examples(dataset, dataset_name: str):
    limit = DATASET_SAMPLE_LIMITS.get(dataset_name)
    if limit is None:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


@dataclass
class AttrConfig:
    model_path: str
    device: str = "cuda:0"
    i_block: int = 100        

def _apply_rotary_pos_emb_local(q, k, cos, sin, unsqueeze_dim=1):
    """Apply RoPE to q,k using provided cos/sin tensors (HF-compatible broadcasting)."""
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    if unsqueeze_dim is not None:
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out, k_out


class GemmaAttributor:
    def __init__(self, cfg: AttrConfig):
        self.cfg = cfg
        self.torch_dtype = torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            # Gemma typically doesn't define a pad token; map to eos
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path, torch_dtype=self.torch_dtype, attn_implementation="eager",
        ).to(cfg.device)
        self.model.eval()
        self.model.config.output_attentions = True 
        self.model.config.use_cache = True
        # print(self.model)
        mcfg = self.model.config.text_config
        self.num_layers = getattr(mcfg, "num_hidden_layers", None) or getattr(mcfg, "n_layer")
        self.num_heads = getattr(mcfg, "num_attention_heads", None) or getattr(mcfg, "n_head")
        self.n_kv_heads = getattr(mcfg, "num_key_value_heads", None) or getattr(mcfg, "n_kv_heads", self.num_heads)
        self.hidden_size = getattr(mcfg, "hidden_size", None) or getattr(mcfg, "n_embd")
        self.repeat_ratio = max(1, self.num_heads // max(1, self.n_kv_heads))
        self.softcap = float(getattr(mcfg, "attn_logit_softcapping", 0.0) or 0.0)
        self.text_model = self.model.model.language_model
        self.layers = self.text_model.layers
        self.embed_tokens = self.text_model.embed_tokens
        self.rope = getattr(self.text_model, "rotary_emb", None) or getattr(self.model.model, "rotary_emb", None)
        # if self.rope is None:
        #     raise AttributeError("Gemma-3:  otary_emb（language_model.rotary_emb）")

    @staticmethod
    def build_prompt(context: str, question: str) -> str:
        return f"Answer the question in no more than five words. Context: {context} Question: {question} Answer:"

    @staticmethod
    def _accumulate_last_token(device, contrib_mats: List[np.ndarray], seq_len: int) -> List[float]:
        P = torch.eye(seq_len, device=device, dtype=torch.bfloat16)
        for C in contrib_mats:
            C_16 = C.to(device, dtype=torch.bfloat16)
            P = C_16 @ P
        last = P[-1, :].squeeze()
        return [float(x) for x in last.tolist()]

    def generate_one_token(self, prompt: str, do_sample: bool = False,
                           temperature: float = 1.0, top_p: float = 1.0) -> Tuple[int, str]:
        device = self.cfg.device
        enc = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = self.model(
                **enc,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
    )
            logits = out.logits[:, -1, :]  # [B,V]
            if do_sample:
                probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    cutoff = (cumsum > top_p).float().argmax(dim=-1, keepdim=True)
                    mask = (torch.arange(sorted_probs.size(-1), device=probs.device)[None, :] <= cutoff).to(probs.dtype)
                    truncated = sorted_probs * mask
                    truncated = truncated / (truncated.sum(dim=-1, keepdim=True) + 1e-12)
                    sampled = torch.multinomial(truncated, num_samples=1)
                    next_id = sorted_idx.gather(1, sampled)
                else:
                    next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
        next_id = next_id.squeeze(0).item()
        next_text = self.tokenizer.decode([next_id], skip_special_tokens=False)
        return next_id, next_text
    
    def _capture_o_proj_refs(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[int, torch.Tensor]:
        outputs_after_o_proj = {}
        outputs_after_mlp   = {}
        outputs_after_post_attention   = {}
        outputs_after_pre_feedforward   = {}
        outputs_after_gate_proj   = {}
        outputs_after_up_proj   = {}
        attn_probs_refs = {}

        def make_hook(idx):
            def hook_fn(module, inp, out):
                outputs_after_o_proj[idx] = out.detach()
            return hook_fn

        def make_hook_mlp(idx):
            def hook_fn(module, inp, out):
                outputs_after_mlp[idx] = out.detach()
            return hook_fn
        
        def make_hook_post_attention(idx):
            def hook_fn(module, inp, out):
                outputs_after_post_attention[idx] = out.detach()
            return hook_fn
        
        def make_hook_pre_feedforward(idx):
            def hook_fn(module, inp, out):
                 outputs_after_pre_feedforward[idx] = out.detach()
            return hook_fn
        
        def make_hook_gate_proj(idx):
            def hook_fn(module, inp, out):
                outputs_after_gate_proj[idx] = out.detach()
            return hook_fn
        
        def make_hook_up_proj(idx):
            def hook_fn(module, inp, out):
                outputs_after_up_proj[idx] = out.detach()
            return hook_fn
    
        hooks = []
        # orig_forwards = []
        for li, block in enumerate(self.layers):
            # hooks.append(block.self_attn.register_forward_hook(make_hook_self_attn(li)))
            hooks.append(block.post_attention_layernorm.register_forward_hook(make_hook_post_attention(li)))
            hooks.append(block.pre_feedforward_layernorm.register_forward_hook(make_hook_pre_feedforward(li)))
            hooks.append(block.mlp.gate_proj.register_forward_hook(make_hook_gate_proj(li)))
            hooks.append(block.mlp.up_proj.register_forward_hook(make_hook_up_proj(li)))
            hooks.append(block.self_attn.o_proj.register_forward_hook(make_hook(li)))
            hooks.append(block.post_feedforward_layernorm.register_forward_hook(make_hook_mlp(li)))
        with torch.no_grad():
            out = self.model(
                **input_tensors,
                output_attentions=True,
                use_cache=True,
                return_dict=True,
            )
        attn_probs_refs = {
                    layer_idx: out.attentions[layer_idx].detach()
                    for layer_idx in range(self.num_layers)
                        }
        v_refs = {}
        for i in range(self.num_layers):
            v_i = out.past_key_values[i][1].detach()
            if v_i.dim() == 4 and v_i.shape[1] != self.n_kv_heads and v_i.shape[2] == self.n_kv_heads:
                v_i = v_i.permute(0, 2, 1, 3).contiguous()
            v_refs[i] = v_i  # [B, KVH, T, d_kv]
        for h in hooks:
            h.remove()
        return outputs_after_o_proj, outputs_after_mlp, outputs_after_post_attention, outputs_after_pre_feedforward, outputs_after_gate_proj, outputs_after_up_proj, attn_probs_refs, v_refs

    def _manual_forward_once_and_calc_matrices(self, input_ids: torch.Tensor,
                                               oproj_refs: Dict[int, torch.Tensor], outputs_after_mlp: Dict[int, torch.Tensor],outputs_after_post_attention, outputs_after_pre_feedforward, outputs_after_gate_proj, outputs_after_up_proj, attn_probs_refs, v_refs, attention_mask = None) -> List[torch.Tensor]:
        device = self.cfg.device
        # model = self.model
        batch_size, seq_len = input_ids.shape
        # base_position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        # base_position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp_(min=0)
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
        base_position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp_(min=0)
        hidden_states = self.embed_tokens(input_ids.to(device))  # [B,T,D]
        # position_embeddings = self.rotary_emb(hidden_states, base_position_ids)
        normalizer = torch.tensor(self.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer       

        contrib_mats = []

        for i, block in enumerate(self.layers):
            hidden_states_ln = block.input_layernorm(hidden_states)

            # q/k/v
            q = block.self_attn.q_proj(hidden_states_ln)  # [B,T,D]
            k = block.self_attn.k_proj(hidden_states_ln)
            v_lin = block.self_attn.v_proj(hidden_states_ln)

            q_head_dim = q.shape[-1] // self.num_heads
            kv_head_dim = k.shape[-1] // self.n_kv_heads

            q = q.view(batch_size, seq_len, self.num_heads, q_head_dim).transpose(1, 2)  # [B,H,T,d]
            k = k.view(batch_size, seq_len, self.n_kv_heads, kv_head_dim).transpose(1, 2)
            v = v_lin.view(batch_size, seq_len, self.n_kv_heads, kv_head_dim).transpose(1, 2)
            q = block.self_attn.q_norm(q)   # [B,H,T,d]
            k = block.self_attn.k_norm(k)   # [B,KVH,T,d]
            # RoPE
            # rope = self.rope(block, base_position_ids)
            q_like = q.transpose(1, 2)[..., :q_head_dim]
            cos, sin = self.rope(q_like, base_position_ids)
            rotary_dim = cos.size(-1)
            q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
            k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
            q_rot, k_rot = _apply_rotary_pos_emb_local(q_rot, k_rot, cos, sin, unsqueeze_dim=1)
            q = torch.cat([q_rot, q_pass], dim=-1)
            k = torch.cat([k_rot, k_pass], dim=-1)
            repeat_ratio = self.num_heads // self.n_kv_heads
            v = v_refs[i]  # 直接使用 HF 计算得到的 v 作为参考
            if repeat_ratio > 1:
                k = k.repeat_interleave(repeat_ratio, dim=1)
                v = v.repeat_interleave(repeat_ratio, dim=1)
            attn_probs = attn_probs_refs[i]

            # o_proj
            W_O_full = block.self_attn.o_proj.weight.view(self.hidden_size, self.num_heads, q_head_dim).permute(1, 0, 2).to(device)  # [H,D,d]
            P = torch.einsum("b h j d, h D d -> b h j D", v, W_O_full)  # [B,H,T,D]
            # total_attn_output = torch.einsum("b h i j, b h j D -> b i D", attn_probs, P)
            attn_out = torch.einsum("b h i j, b h j d -> b h i d", attn_probs, v)           # [B,H,T,kv_head_dim]
            attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1) # [B,T,H*kv_head_dim]
            total_attn_output_std = torch.nn.functional.linear(attn_out, block.self_attn.o_proj.weight, bias=None)
            mat_accum = torch.zeros(batch_size, seq_len, seq_len, device=device, dtype=torch.float32)
            # Y_full = hidden_states + total_attn_output  # [B,T,D]
            attn_normed = block.post_attention_layernorm(oproj_refs[i])   # LN: post-attn
            # print("post_attention_diff (std path):", float(torch.norm(outputs_after_post_attention[i] - attn_normed)))
            Y_full = hidden_states + attn_normed  # [B,T,D]
            step = max(1, int(self.cfg.i_block))
            for i0 in range(0, seq_len, step):
                i1 = min(i0 + step, seq_len)
                Ti = i1 - i0

                A_blk = attn_probs[:, :, i0:i1, :]                   # [B,H,Ti,T]
                contrib_blk = torch.einsum("b h i j, b h j D -> b i j D", A_blk, P)  # [B,Ti,T,D]
                residual_blk = hidden_states[:, i0:i1, :]            # [B,Ti,D]

                idx = torch.arange(Ti, device=device)  # positions within the block
                contrib_blk[:, idx, i0 + idx, :] += residual_blk

                Y_blk = Y_full[:, i0:i1, :]  # [B,Ti,D]

                dists = (Y_blk[:, :, None, :] - contrib_blk).abs().sum(dim=-1)   # [B,Ti,T]
                y_norm = Y_blk.abs().sum(dim=-1, keepdim=True)                   # [B,Ti,1]
                numerators = torch.clamp(-dists + y_norm, min=0.0)
                denom = numerators.sum(dim=-1, keepdim=True) + 1e-4
                mat_blk = numerators / denom                                     # [B,Ti,T]

                mat_accum[:, i0:i1, :] = mat_blk

                del A_blk, contrib_blk, residual_blk, Y_blk
                torch.cuda.empty_cache()
            # print("pre_feedforward_diff (std path):", float(torch.norm(outputs_after_pre_feedforward[i] - hidden_states_after_attn)))
            mlp_in = outputs_after_pre_feedforward[i]
            gate = block.mlp.gate_proj(mlp_in)                       # [B,T,14336]
            up   = block.mlp.up_proj(mlp_in)                         # [B,T,14336]
            hidden = block.mlp.act_fn(gate) * up                
            mlp_out = block.mlp.down_proj(hidden)               # [B,T,3584]
            mlp_out = block.post_feedforward_layernorm(mlp_out) # LN: post-FF
 
            hidden_states = (
            hidden_states
            + outputs_after_post_attention[i]
            + outputs_after_mlp[i]
        ) 
            contrib_mats.append(mat_accum[0])     

            del q, k, v, v_lin, P, total_attn_output_std, mlp_out
            torch.cuda.empty_cache()

        return contrib_mats
