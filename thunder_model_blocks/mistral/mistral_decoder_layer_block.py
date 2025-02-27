import sys
import torch

from thunder_model_blocks.utils import runner
from thunder_model_blocks.mistral import mistral_config
from transformers.cache_utils import DynamicCache
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

class MyModel(torch.nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.model = MistralDecoderLayer(config, 0)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        cache_position,
        position_ids,
        ) :
        kwargs = {}
        out = self.model(hidden_states=hidden_states,
                         attention_mask=None,
                         position_ids=position_ids,
                         past_key_value=DynamicCache(),
                         use_cache=True,
                         cache_position=cache_position,
                         position_embeddings=position_embeddings,
                         **kwargs
                         )
        return out

if __name__ == "__main__":
    cfg = mistral_config.config()

    def inputs(dtype, batch_size=cfg.batch_size, seq_len=cfg.seq_len, packed_seq_fn=None):
        args = {
            "hidden_states": torch.randn(batch_size, seq_len, cfg.hidden_size, device='cuda', dtype=dtype, requires_grad=True),
            "position_embeddings": (
                torch.randn(batch_size, seq_len, cfg.head_dim, device='cuda', dtype=dtype, requires_grad=False),
                torch.randn(batch_size, seq_len, cfg.head_dim, device='cuda', dtype=dtype, requires_grad=False),
            ),
            "cache_position": torch.arange(seq_len, device='cuda'),
            "position_ids": torch.arange(seq_len, device='cuda'),
        }
        return args
    
    def grads(dtype, batch_size=cfg.batch_size, seq_len=cfg.seq_len):
        grad = torch.randn(batch_size, seq_len, cfg.hidden_size, device='cuda', dtype=dtype, requires_grad=False)
        return grad

    runner.run(sys.argv, cfg.name_or_path, cfg, MyModel, inputs, module_has_loss=False, grad_fn=grads)
