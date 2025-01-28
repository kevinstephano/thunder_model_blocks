import json
import sys
import torch
from torch import nn
from typing import Tuple
from thunder_model_blocks.utils import runner
from transformers.models.phi3 import Phi3Config

"""
# Example to download model and config
from transformers import AutoConfig, AutoModel
model = AutoModel.from_pretrained("microsoft/Phi-3.5-mini-instruct")
config = AutoConfig.from_pretrained("microsoft/Phi-3.5-mini-instruct")
"""

phi35_cfg_str = r'''{
  "_name_or_path": "microsoft/Phi-3.5-mini-instruct",
  "architectures": [
    "Phi3ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "auto_map": {
    "AutoConfig": "microsoft/Phi-3.5-mini-instruct--configuration_phi3.Phi3Config",
    "AutoModelForCausalLM": "microsoft/Phi-3.5-mini-instruct--modeling_phi3.Phi3ForCausalLM"
  },
  "bos_token_id": 1,
  "embd_pdrop": 0.0,
  "eos_token_id": 32000,
  "hidden_act": "silu",
  "hidden_size": 3072,
  "initializer_range": 0.02,
  "intermediate_size": 8192,
  "max_position_embeddings": 131072,
  "model_type": "phi3",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "original_max_position_embeddings": 4096,
  "pad_token_id": 32000,
  "resid_pdrop": 0.0,
  "rms_norm_eps": 1e-05,
  "rope_scaling": {
    "long_factor": [
      1.0800000429153442,
      1.1100000143051147,
      1.1399999856948853,
      1.340000033378601,
      1.5899999141693115,
      1.600000023841858,
      1.6200000047683716,
      2.620000123977661,
      3.2300000190734863,
      3.2300000190734863,
      4.789999961853027,
      7.400000095367432,
      7.700000286102295,
      9.09000015258789,
      12.199999809265137,
      17.670000076293945,
      24.46000099182129,
      28.57000160217285,
      30.420001983642578,
      30.840002059936523,
      32.590003967285156,
      32.93000411987305,
      42.320003509521484,
      44.96000289916992,
      50.340003967285156,
      50.45000457763672,
      57.55000305175781,
      57.93000411987305,
      58.21000289916992,
      60.1400032043457,
      62.61000442504883,
      62.62000274658203,
      62.71000289916992,
      63.1400032043457,
      63.1400032043457,
      63.77000427246094,
      63.93000411987305,
      63.96000289916992,
      63.970001220703125,
      64.02999877929688,
      64.06999969482422,
      64.08000183105469,
      64.12000274658203,
      64.41000366210938,
      64.4800033569336,
      64.51000213623047,
      64.52999877929688,
      64.83999633789062
    ],
   "short_factor": [
      1.0,
      1.0199999809265137,
      1.0299999713897705,
      1.0299999713897705,
      1.0499999523162842,
      1.0499999523162842,
      1.0499999523162842,
      1.0499999523162842,
      1.0499999523162842,
      1.0699999332427979,
      1.0999999046325684,
      1.1099998950958252,
      1.1599998474121094,
      1.1599998474121094,
      1.1699998378753662,
      1.2899998426437378,
      1.339999794960022,
      1.679999828338623,
      1.7899998426437378,
      1.8199998140335083,
      1.8499997854232788,
      1.8799997568130493,
      1.9099997282028198,
      1.9399996995925903,
      1.9899996519088745,
      2.0199997425079346,
      2.0199997425079346,
      2.0199997425079346,
      2.0199997425079346,
      2.0199997425079346,
      2.0199997425079346,
      2.0299997329711914,
      2.0299997329711914,
      2.0299997329711914,
      2.0299997329711914,
      2.0299997329711914,
      2.0299997329711914,
      2.0299997329711914,
      2.0299997329711914,
      2.0299997329711914,
      2.0799996852874756,
      2.0899996757507324,
      2.189999580383301,
      2.2199995517730713,
      2.5899994373321533,
      2.729999542236328,
      2.749999523162842,
      2.8399994373321533
    ],
    "type": "longrope"
  },
  "rope_theta": 10000.0,
  "sliding_window": 262144,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.46.3",
  "use_cache": true,
  "vocab_size": 32064
}'''

class Phi3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000.0, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class HfPhi3Rope(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Phi3Config):
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.original_max_position_embeddings = config.original_max_position_embeddings
        self.rope_theta = config.rope_theta
        self.rope_scaling = config.rope_scaling
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.rotary_emb = Phi3RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        qkv: torch.Tensor,
        position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        past_key_value = None
        bsz, q_len, _ = qkv.size()

        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            assert False
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            assert False

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        return query_states, key_states, value_states

phi3_cfg = Phi3Config.from_dict(json.loads(phi35_cfg_str))
phi3_cfg.batch_size = 1
phi3_cfg.seq_len = 8192
configs = {}
configs[phi3_cfg.name_or_path] = phi3_cfg

if __name__ == "__main__":
    #print(configs["Phi3"].to_json_string(use_diff=False))
    for name,cfg in configs.items():
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        def inputs():
            args = {
                "qkv": torch.randn(cfg.batch_size, cfg.seq_len, cfg.num_attention_heads * head_dim + 2 * (cfg.num_key_value_heads * head_dim), device='cuda', dtype=torch.bfloat16, requires_grad=True),
                "position_ids": torch.arange(0, cfg.seq_len, device='cuda').unsqueeze(0),
            }
            return args
        def grads():
            grad = torch.randn(cfg.batch_size, cfg.num_attention_heads, cfg.seq_len, head_dim, device='cuda', dtype=torch.bfloat16, requires_grad=False)
            return grad
 
        model = HfPhi3Rope(cfg).cuda().bfloat16()
        runner.run(sys.argv, name, cfg.batch_size, cfg.seq_len, model, inputs, False, grads)
