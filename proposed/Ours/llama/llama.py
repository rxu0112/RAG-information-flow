import time
import json
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

@dataclass
class AttrConfig:
    model_path: str
    device: str = "cuda:0"
    i_block: int = 10       

class LlamaAttributor:
    def __init__(self, cfg: AttrConfig):
        self.cfg = cfg
        self.torch_dtype = torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path, torch_dtype=self.torch_dtype
        ).to(cfg.device)
        self.model.eval()

        mcfg = self.model.config
        self.num_layers = mcfg.num_hidden_layers
        self.num_heads = mcfg.num_attention_heads
        self.n_kv_heads = mcfg.num_key_value_heads
        self.hidden_size = mcfg.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.repeat_ratio = self.num_heads // self.n_kv_heads
        self.softcap = float(getattr(mcfg, "attn_logit_softcapping", 0.0) or 0.0)

    @staticmethod
    def build_prompt(context: str, question: str) -> str:
        return f"Answer the question in no more than five words. Context: {context} Question: {question} Answer:"
        # return f"Answer in a full sentence. Context: {context} Question: {question} Answer:"

    def _get_rope(self, block):
        rope = getattr(block.self_attn, "rotary_emb", None)
        if rope is None:
            rope = getattr(self.model.model, "rotary_emb", None)
        if rope is None:
            raise AttributeError("No rotary_emb found on block.self_attn or model.model")
        return rope

    def _capture_o_proj_refs(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[int, torch.Tensor]:
        outputs_after_o_proj = {}

        def make_hook(idx):
            def hook_fn(module, inp, out):
                outputs_after_o_proj[idx] = out.detach()
            return hook_fn

        hooks = []
        for li, block in enumerate(self.model.model.layers):
            hooks.append(block.self_attn.o_proj.register_forward_hook(make_hook(li)))

        _ = self.model(**input_tensors)  # 触发 hook
        for h in hooks:
            h.remove()
        return outputs_after_o_proj

    @staticmethod
    def _accumulate_last_token(device, contrib_mats: List[np.ndarray], seq_len: int) -> List[float]:
        # P = np.eye(seq_len)
        P = torch.eye(seq_len, device=device, dtype=torch.bfloat16)
        for C in contrib_mats:
            C_16 = C.to(device, dtype=torch.bfloat16)
            P = C_16 @ P
        # last = P[-1, :]
        last = P[-1, :].squeeze()
        # last = last.flatten()
        return [float(x) for x in last.tolist()]
    

    def generate_one_token(self, prompt: str, do_sample: bool = False,
                           temperature: float = 1.0, top_p: float = 1.0) -> Tuple[int, str]:
        device = self.cfg.device
        enc = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = self.model(**enc)
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


    def _manual_forward_once_and_calc_matrices(self, input_ids: torch.Tensor,
                                            oproj_refs: Dict[int, torch.Tensor]) -> List[torch.Tensor]:
        device = self.cfg.device
        model = self.model
        batch_size, seq_len = input_ids.shape
        base_position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)

        hidden_states = model.model.embed_tokens(input_ids.to(device))  # [B,T,D]

        contrib_mats = []

        for block in model.model.layers:
            hidden_states_ln = block.input_layernorm(hidden_states)

            # q/k/v
            q = block.self_attn.q_proj(hidden_states_ln)  # [B,T,D]
            k = block.self_attn.k_proj(hidden_states_ln)
            v_lin = block.self_attn.v_proj(hidden_states_ln)

            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,T,d]
            k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = v_lin.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

            # RoPE
            rope = self._get_rope(block)
            q_like = q.transpose(1, 2)[..., :self.head_dim]
            cos, sin = rope(q_like, base_position_ids)
            rotary_dim = cos.size(-1)
            q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
            k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin, unsqueeze_dim=1)
            q = torch.cat([q_rot, q_pass], dim=-1)
            k = torch.cat([k_rot, k_pass], dim=-1)

            if self.repeat_ratio > 1:
                k = k.repeat_interleave(self.repeat_ratio, dim=1)
                v = v.repeat_interleave(self.repeat_ratio, dim=1)

            # scores + mask + softmax
            attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
            if self.softcap > 0.0:
                attn_scores = torch.tanh(attn_scores / self.softcap) * self.softcap
            mask = torch.triu(
                torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device, dtype=attn_scores.dtype),
                diagonal=1
            )
            attn_scores = attn_scores + mask
            attn_probs = F.softmax(attn_scores, dim=-1, dtype=self.torch_dtype)  # [B,H,T,T]

            # o_proj
            W_O_full = block.self_attn.o_proj.weight.view(self.hidden_size, self.num_heads, self.head_dim).permute(1, 0, 2).to(device)  # [H,D,d]
            P = torch.einsum("b h j d, h D d -> b h j D", v, W_O_full)  # [B,H,T,D]
            total_attn_output = torch.einsum("b h i j, b h j D -> b i D", attn_probs, P)  # [B,T,D]

            mat_accum = torch.zeros(batch_size, seq_len, seq_len, device=device, dtype=torch.float32)
            Y_full = hidden_states + total_attn_output  # [B,T,D]

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
                y_norm = Y_blk.abs().sum(dim=-1, keepdim=True)                    # [B,Ti,1]
                numerators = torch.clamp(-dists + y_norm, min=0.0)
                denom = numerators.sum(dim=-1, keepdim=True) + 1e-4
                mat_blk = numerators / denom                                      # [B,Ti,T]

                mat_accum[:, i0:i1, :] = mat_blk

                del A_blk, contrib_blk, residual_blk, Y_blk
                torch.cuda.empty_cache()
            hidden_states = hidden_states + total_attn_output
            attn_after_res = hidden_states.detach()

            # MLP
            hidden_states_ln2 = block.post_attention_layernorm(hidden_states)
            gate = block.mlp.gate_proj(hidden_states_ln2)
            up = block.mlp.up_proj(hidden_states_ln2)
            mlp_output = block.mlp.down_proj(F.silu(gate) * up)
            hidden_states = hidden_states + mlp_output

            contrib_mats.append(mat_accum[0])     # [T,T]

            del q, k, v, v_lin, P, total_attn_output, mlp_output
            torch.cuda.empty_cache()

        return contrib_mats
