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
        position_ids,
        ) :
        kwargs = {"position_ids": position_ids, "output_attentions": False, "use_cache": True}
        out = self.model(hidden_states=hidden_states,
                         position_embeddings=position_embeddings,
                         attention_mask=None,
                         past_key_value=DynamicCache(),
                         cache_position=cache_position,
                         **kwargs
                         )
        return out

if __name__ == "__main__":
    config = qwen2_config.config()
    config.lora = True
    configs = {config.name_or_path: config}

    for name,cfg in configs.items():
        attn_hidden_size = cfg.hidden_size // cfg.num_attention_heads
        def inputs(dtype, batch_size=cfg.batch_size, seq_len=cfg.seq_len):
            hidden_states = torch.randn(batch_size, seq_len, cfg.hidden_size, device='cuda', dtype=dtype, requires_grad=True)
            position_embeddings = (
                    torch.randn(cfg.batch_size, cfg.seq_len, attn_hidden_size, device='cuda', dtype=dtype, requires_grad=False),
                    torch.randn(cfg.batch_size, cfg.seq_len, attn_hidden_size, device='cuda', dtype=dtype, requires_grad=False),
                    )
            cache_position = torch.arange(cfg.seq_len, device='cuda')
            position_ids = torch.arange(cfg.seq_len, device='cuda')
            return {"hidden_states": hidden_states,
                    "position_embeddings": position_embeddings,
                    "cache_position": cache_position,
                    "position_ids": position_ids,
                    }
        
        def grads(dtype, batch_size=cfg.batch_size, seq_len=cfg.seq_len):
            grad = torch.randn(batch_size, seq_len, cfg.hidden_size, device='cuda', dtype=dtype, requires_grad=False)
            return grad

        runner.run(sys.argv, name, cfg, MyModel, inputs, module_has_loss=False, grad_fn=grads)
