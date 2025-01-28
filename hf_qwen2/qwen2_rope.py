import json
import sys
import sys
import torch
from torch import nn
from transformers.models.qwen2 import Qwen2Config
from typing import Tuple
from thunder_model_blocks.utils import runner

'''
# Example to download model and config
from transformers import AutoConfig, AutoModel
model = AutoModel.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
config = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
'''

qwen_cfg_str = r'''{
  "_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 3584,
  "initializer_range": 0.02,
  "intermediate_size": 18944,
  "max_position_embeddings": 32768,
  "max_window_layers": 28,
  "model_type": "qwen2",
  "num_attention_heads": 28,
  "num_hidden_layers": 28,
  "num_key_value_heads": 4,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "sliding_window": null,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.43.3",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 152064
}
'''

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

class Qwen2Rope(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

    def forward(
        self,
        query_in_states: torch.Tensor,
        key_in_states: torch.Tensor,
        value_in_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        past_key_value = None
        bsz, q_len, _ = query_in_states.size()

        query_states = query_in_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_in_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_in_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            assert False

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        return query_states, key_states, value_states

qwen2_cfg = Qwen2Config.from_dict(json.loads(qwen_cfg_str))
qwen2_cfg.batch_size = 1
qwen2_cfg.seq_len = 4096
configs = {}
configs[qwen2_cfg.name_or_path] = qwen2_cfg

if __name__ == "__main__":
    for name,cfg in configs.items():
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        def inputs():
            args = {
                "query_in_states": torch.randn(cfg.batch_size, cfg.seq_len, cfg.num_attention_heads * head_dim, device='cuda', dtype=torch.bfloat16, requires_grad=True),
                "key_in_states": torch.randn(cfg.batch_size, cfg.seq_len, cfg.num_key_value_heads * head_dim, device='cuda', dtype=torch.bfloat16, requires_grad=True),
                "value_in_states": torch.randn(cfg.batch_size, cfg.seq_len, cfg.num_key_value_heads * head_dim, device='cuda', dtype=torch.bfloat16, requires_grad=True),
                "cos": torch.randn(cfg.batch_size, cfg.seq_len, head_dim, device='cuda', dtype=torch.bfloat16, requires_grad=True),
                "sin": torch.randn(cfg.batch_size, cfg.seq_len, head_dim, device='cuda', dtype=torch.bfloat16, requires_grad=True),
            }
            return args
        def grads():
            grad = torch.randn(cfg.batch_size, cfg.num_attention_heads, cfg.seq_len, head_dim, device='cuda', dtype=torch.bfloat16, requires_grad=False)
            return grad
 
        model = Qwen2Rope(cfg).cuda().bfloat16()
        runner.run(sys.argv, name, cfg.batch_size, cfg.seq_len, model, inputs, False, grads)
