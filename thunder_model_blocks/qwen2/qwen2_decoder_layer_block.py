import sys
import torch

from thunder_model_blocks.utils import runner
from thunder_model_blocks.qwen2 import qwen2_config
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

class MyModel(torch.nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.model = Qwen2DecoderLayer(config, 0)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        cache_position,
        attention_mask,
        position_ids,
        ) :
        kwargs = {"output_attentions": False}
        out = self.model(hidden_states=hidden_states,
                         attention_mask=attention_mask,
                         position_ids=position_ids,
                         past_key_value=DynamicCache(),
                         use_cache=True,
                         cache_position=cache_position,
                         position_embeddings=position_embeddings,
                         **kwargs
                         )
        return out

if __name__ == "__main__":
    cfg = qwen2_config.config()

    attn_hidden_size = cfg.hidden_size // cfg.num_attention_heads
    def inputs(dtype, batch_size=cfg.batch_size, seq_len=cfg.seq_len, packed_seq_fn=None):
        attention_mask = None
        position_ids = torch.arange(cfg.seq_len, device='cuda'),
        if packed_seq_fn is not None:
            attention_mask, position_ids = packed_seq_fn(batch_size=batch_size, seq_len=seq_len)
        args = {
            "hidden_states": torch.randn(batch_size, seq_len, cfg.hidden_size, device='cuda', dtype=dtype, requires_grad=True),
            "position_embeddings": (
                    torch.randn(cfg.batch_size, cfg.seq_len, attn_hidden_size, device='cuda', dtype=dtype, requires_grad=False),
                    torch.randn(cfg.batch_size, cfg.seq_len, attn_hidden_size, device='cuda', dtype=dtype, requires_grad=False),
                    ),
            "cache_position": torch.arange(cfg.seq_len, device='cuda'),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        return args
    
    def grads(dtype, batch_size=cfg.batch_size, seq_len=cfg.seq_len):
        grad = torch.randn(batch_size, seq_len, cfg.hidden_size, device='cuda', dtype=dtype, requires_grad=False)
        return grad
    
    runner.run(sys.argv, cfg.name_or_path, cfg, MyModel, inputs, module_has_loss=False, grad_fn=grads)
