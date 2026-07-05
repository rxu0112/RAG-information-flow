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
    dtype: str = "fp16"          # "fp16" | "bf16" | "fp32"
    metric: str = "manhattan"    # "manhattan" | "dot"
    i_block: int = 10          

def _dtype_from_str(s: str):
    if s == "fp16": return torch.float16
    if s == "bf16": return torch.bfloat16
    return torch.float32

class LlamaAttributor:
    def __init__(self, cfg: AttrConfig):
        self.cfg = cfg
        self.torch_dtype = _dtype_from_str(cfg.dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path, torch_dtype=self.torch_dtype
        ).to(cfg.device)
        self.model.eval()

        mcfg = self.model.config
        if hasattr(mcfg, 'text_config'):
            mcfg = mcfg.text_config

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
       

    @staticmethod
    def append_token_to_prompt(prompt: str, token_text: str) -> str:
        return prompt + token_text

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

        _ = self.model(**input_tensors)  
        for h in hooks:
            h.remove()
        return outputs_after_o_proj

    def get_last_hidden(self, prompt: str) -> torch.Tensor:
        device = self.cfg.device
        enc = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
            hidden = out.hidden_states[-1]  # [1,T,D]


        return hidden
    
    def _manual_forward_once(self, input_ids: torch.Tensor,
                             oproj_refs: Dict[int, torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        device = self.cfg.device
        model = self.model
        batch_size, seq_len = input_ids.shape
        base_position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)

        hidden_states = model.model.embed_tokens(input_ids.to(device))  # [B,T,D]

        attn_res_output_blocks = []
        attn_after_residual_blocks = []

        for li, block in enumerate(model.model.layers):
            # layer input
            # hidden_states = hidden_states.to(device, dtype=next(model.parameters()).dtype)
            hidden_states_ln = block.input_layernorm(hidden_states)
            # position_ids = base_position_ids.to(device)

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
            attn_probs = F.softmax(attn_scores, dim=-1, dtype=self.torch_dtype)  # [B,H,T,T] (fp32)

            W_O_full = block.self_attn.o_proj.weight.view(self.hidden_size, self.num_heads, self.head_dim).permute(1, 0, 2).to(device)  # [H,D,d]
            P = torch.einsum("b h j d, h D d -> b h j D", v, W_O_full)  # [B,H,T,D]
            total_attn_output = torch.einsum("b h i j, b h j D -> b i D", attn_probs, P)  # [B,T,D]

            ref = oproj_refs[li].to(device)
            diff = (total_attn_output - ref).abs().max().item()
            step = max(1, int(self.cfg.i_block))
            contrib_layer_blk = []
            for i0 in range(0, seq_len, step):
                i1 = min(i0 + step, seq_len)
                Ti = i1 - i0
                A_blk = attn_probs[:, :, i0:i1, :]                  # [B,H,Ti,T]
                contrib_blk = torch.einsum("b h i j, b h j D -> b i j D", A_blk, P)  # [B,Ti,T,D]
                residual_blk = hidden_states[:, i0:i1, :]           # [B,Ti,D]

                batch_idx = torch.arange(contrib_blk.size(0), device=contrib_blk.device)[:, None]
                time_idx  = torch.arange(Ti, device=contrib_blk.device)[None, :]
                seq_idx   = (i0 + torch.arange(Ti, device=contrib_blk.device))[None, :]
                contrib_blk[batch_idx, time_idx, seq_idx, :] += residual_blk

                # print(res_out_full.dtype)
                # res_out_full_cpu[:, i0:i1, :, :] = res_out_blk.detach().to("cpu", dtype=self.torch_dtype)
                contrib_layer_blk.append(contrib_blk.to("cpu"))
                del A_blk, residual_blk
                torch.cuda.empty_cache()

            hidden_states = hidden_states + total_attn_output
            attn_after_res = hidden_states.detach()
            # print(attn_after_res.dtype)
            # attn_after_res_cpu = hidden_states.detach().to("cpu", dtype=self.torch_dtype)

            # MLP
            hidden_states_ln2 = block.post_attention_layernorm(hidden_states)
            gate = block.mlp.gate_proj(hidden_states_ln2)
            up = block.mlp.up_proj(hidden_states_ln2)
            mlp_output = block.mlp.down_proj(F.silu(gate) * up)
            hidden_states = hidden_states + mlp_output

            attn_res_output_blocks.append(contrib_layer_blk)
            attn_after_residual_blocks.append(attn_after_res)

            del q, k, v, v_lin, P, total_attn_output, mlp_output
            torch.cuda.empty_cache()

        return attn_res_output_blocks, attn_after_residual_blocks

    def _contrib_matrices(self, attn_res_output_blocks: List[torch.Tensor],
                          attn_after_residual_blocks: List[torch.Tensor]) -> List[np.ndarray]:
        device = self.cfg.device 
        contrib_mats = []
        for li in range(self.num_layers):
            res_full_gpu = torch.cat(attn_res_output_blocks[li], dim=1).to(device)  # [B,T,T,D]
            Y_full  = attn_after_residual_blocks[li]     # [B,T,D]
            Y_full_gpu = Y_full.to(device)
            assert res_full_gpu.ndim == 4 and Y_full.ndim == 3
            B, T, Tp1, D = res_full_gpu.shape
            C = res_full_gpu[0, :, :T, :]  # [T, T, D]
            Y = Y_full_gpu[0, :, :]       # [T, D]


            if self.cfg.metric == "manhattan":
                # dists = (Y[:, None, :] - C).abs().sum(dim=-1)     # [T, T]
                dists = (Y[:, None, :] - C).abs().sum(dim=-1)     # [T, T]
                y_norm = Y.abs().sum(dim=-1, keepdim=True)        # [T, 1]
                numerators = torch.clamp(-dists + y_norm, min=0.0)
                denom = numerators.sum(dim=-1, keepdim=True) + 1e-4
                mat = numerators / denom                           # [T, T]
            elif self.cfg.metric == "dot":
                C = C.to(torch.float32)  
                Y = Y.to(torch.float32) 
                dot = (C * Y[:, None, :]).sum(dim=-1)
                y_norm2 = (Y.pow(2).sum(dim=-1, keepdim=True))

                mat = dot / y_norm2
            else:
                raise ValueError(f"Unknown metric: {self.cfg.metric}")
            contrib_mats.append(mat)
        return contrib_mats

    @staticmethod
    def _accumulate_last_token(device, contrib_mats: List[np.ndarray], seq_len: int) -> List[float]:
        P = torch.eye(seq_len, device=device, dtype=torch.bfloat16)
        for C in contrib_mats:
            C_16 = C.to(device, dtype=torch.bfloat16)
            P = C_16 @ P
        last = P[-1, :].squeeze()
        return [float(x) for x in last.tolist()]
    
    @staticmethod
    def _accumulate_last_token2(contrib_mats: List[np.ndarray], seq_len: int) -> List[float]:
        P = np.eye(seq_len-2)
        for C in contrib_mats:
            t = C[:-2, :-2]
            P = t @ P
        last = P[-1, :]
        return [float(x) for x in last.tolist()]
    
    @staticmethod
    def _accumulate_last_token_layerwise(contrib_mats: List[np.ndarray], seq_len: int) -> List[float]:
        layerwise_mats = []
        P = np.eye(seq_len)
        for C in contrib_mats:
            P = C @ P
            last = P[-1, :]
            result = [float(x) for x in last.tolist()]
            layerwise_mats.append(result)
        # last = P[-1, :]
        return layerwise_mats

    def explain_once(self, prompt: str) -> Tuple[List[str], List[float]]:
        device = self.cfg.device
        enc = self.tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        input_ids = enc["input_ids"]
        tokens = [self.tokenizer.decode([tid], skip_special_tokens=False) for tid in input_ids[0].tolist()]

        oproj_refs = self._capture_o_proj_refs(enc)
        attn_res_blocks, attn_after_blocks = self._manual_forward_once(input_ids, oproj_refs)
        mats = self._contrib_matrices(attn_res_blocks, attn_after_blocks)
        contributions = self._accumulate_last_token(mats, input_ids.shape[1])

        return tokens, contributions

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

    def explain_after_appending_one_token(self, prompt: str,
                                          do_sample: bool = False,
                                          temperature: float = 1.0, top_p: float = 1.0
                                          ) -> Dict[str, Any]:
        next_id, next_txt = self.generate_one_token(prompt, do_sample, temperature, top_p)
        if self.tokenizer.eos_token_id is not None and next_id == self.tokenizer.eos_token_id:
            next_txt = "."

        extended_prompt = self.append_token_to_prompt(prompt, next_txt)
        tokens2, contrib2 = self.explain_once(extended_prompt)
        return {
            "generated_token": next_txt,
            "extended_tokens": tokens2,
            "extended_contributions": contrib2,
            "extended_prompt": extended_prompt
        }

    def _manual_forward_once_and_calc_matrices(self, input_ids: torch.Tensor,
                                            oproj_refs: Dict[int, torch.Tensor]) -> List[torch.Tensor]:
        device = self.cfg.device
        model = self.model
        batch_size, seq_len = input_ids.shape
        base_position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)

        hidden_states = model.model.embed_tokens(input_ids.to(device))  # [B,T,D]

        contrib_mats = []

        for li, block in enumerate(model.model.layers):
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

            # contrib = torch.zeros(batch_size, seq_len, seq_len, device=device, dtype=torch.float32)
            mat_accum = torch.zeros(batch_size, seq_len, seq_len, device=device, dtype=torch.float32)
            Y_full = hidden_states + total_attn_output  # [B,T,D]

            step = max(1, int(self.cfg.i_block))
            for i0 in range(0, seq_len, step):
                i1 = min(i0 + step, seq_len)
                Ti = i1 - i0

                A_blk = attn_probs[:, :, i0:i1, :]                   # [B,H,Ti,T]
                contrib_blk = torch.einsum("b h i j, b h j D -> b i j D", A_blk, P)  # [B,Ti,T,D]
                residual_blk = hidden_states[:, i0:i1, :]            # [B,Ti,D]

                batch_idx = torch.arange(batch_size, device=device)[:, None]
                time_idx  = torch.arange(Ti, device=device)[None, :]
                seq_idx   = (i0 + torch.arange(Ti, device=device))[None, :]
                contrib_blk[batch_idx, time_idx, seq_idx, :] += residual_blk

                Y_blk = Y_full[:, i0:i1, :]  # [B,Ti,D]

                if self.cfg.metric == "manhattan":
                    dists = (Y_blk[:, :, None, :] - contrib_blk).abs().sum(dim=-1)   # [B,Ti,T]
                    y_norm = Y_blk.abs().sum(dim=-1, keepdim=True)                  # [B,Ti,1]
                    numerators = torch.clamp(-dists + y_norm, min=0.0)
                    denom = numerators.sum(dim=-1, keepdim=True) + 1e-4
                    mat_blk = numerators / denom                                    # [B,Ti,T]
                elif self.cfg.metric == "dot":
                    contrib_blk = contrib_blk.to(torch.bfloat16)
                    Y_blk = Y_blk.to(torch.bfloat16)
                    dot = (contrib_blk * Y_blk[:, :, None, :]).sum(dim=-1)          # [B,Ti,T]
                    y_norm2 = (Y_blk.pow(2).sum(dim=-1, keepdim=True))              # [B,Ti,1]
                    mat_blk = dot / y_norm2
                else:
                    raise ValueError(f"Unknown metric: {self.cfg.metric}")

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
    
def save_contrib_mats(contrib_mats, save_path="contrib_all_layers.json"):
    mats_list = [mat.detach().cpu().numpy().tolist() for mat in contrib_mats]

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(mats_list, f)

    print(f"✅ Saved {len(mats_list)} layers' contrib matrices to {save_path}")