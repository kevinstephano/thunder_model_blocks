import sys
import torch
from typing import Tuple

from thunder_model_blocks.utils import runner
from thunder_model_blocks.llama import llama_3_8B_Instruct_config

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
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

class LlamaRope(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads


    def forward(
        self,
        query_in_states: torch.Tensor,
        key_in_states: torch.Tensor,
        value_in_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = query_in_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = query_in_states.view(hidden_shape).transpose(1, 2)
        key_states = key_in_states.view(hidden_shape).transpose(1, 2)
        value_states = value_in_states.view(hidden_shape).transpose(1, 2)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # NOTE: I added this to make the loss work, it is minor!
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        return query_states, key_states, value_states

if __name__ == "__main__":
    config = llama_3_8B_Instruct_config.config()
    configs = {config.name_or_path: config}

    for name,cfg in configs.items():
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        def inputs(dtype, batch_size=cfg.batch_size, seq_len=cfg.seq_len):
            args = {
                "query_in_states": torch.randn(batch_size, seq_len, head_dim * cfg.num_attention_heads, device='cuda', dtype=dtype, requires_grad=True),
                "key_in_states": torch.randn(batch_size, seq_len, head_dim * cfg.num_key_value_heads, device='cuda', dtype=dtype, requires_grad=True),
                "value_in_states": torch.randn(batch_size, seq_len, head_dim * cfg.num_key_value_heads, device='cuda', dtype=dtype, requires_grad=True),
                "cos": torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=dtype, requires_grad=True),
                "sin": torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=dtype, requires_grad=True),
            }
            return args
        def grads(dtype, batch_size=cfg.batch_size, seq_len=cfg.seq_len):
            grad = torch.randn(batch_size, cfg.num_attention_heads, seq_len, head_dim, device='cuda', dtype=dtype, requires_grad=False)
            return grad
 
        runner.run(sys.argv, name, cfg, LlamaRope, inputs, module_has_loss=False, grad_fn=grads)
