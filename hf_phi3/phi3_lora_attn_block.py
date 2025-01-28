import torch
from torch import nn
import sys
from thunder_model_blocks.utils import runner
from thunder_model_blocks.utils.lora import patch_linear_module
#from nemo.collections.llm.peft.lora import patch_linear_module

from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.phi3.modeling_phi3 import Phi3Attention

config = AutoConfig.from_pretrained("microsoft/Phi-3.5-mini-instruct")

config.batch_size = 1
config.seq_len = 8192
config._attn_implementation = "sdpa"
configs = {}
configs[config.name_or_path] = config

class MyModel(torch.nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.model = Phi3Attention(config, 0)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        cache_position,
        position_ids,
        ) :
        kwargs = {}
        out,_ = self.model(hidden_states=hidden_states,
                         attention_mask=None,
                         position_ids=position_ids,
                         past_key_value=DynamicCache(),
                         use_cache=True,
                         cache_position=cache_position,
                         position_embeddings=position_embeddings,
                         **kwargs
                         )
        return (out,)

if __name__ == "__main__":
    for name,cfg in configs.items():
        attn_hidden_size = cfg.hidden_size // cfg.num_attention_heads
        def inputs():
            hidden_states = torch.randn(cfg.batch_size, cfg.seq_len, cfg.hidden_size, device='cuda', dtype=torch.bfloat16, requires_grad=True)
            position_embeddings = (
                    torch.randn(cfg.batch_size, cfg.seq_len, attn_hidden_size, device='cuda', dtype=torch.bfloat16, requires_grad=False),
                    torch.randn(cfg.batch_size, cfg.seq_len, attn_hidden_size, device='cuda', dtype=torch.bfloat16, requires_grad=False),
                    )
            cache_position = torch.arange(cfg.seq_len, device='cuda')
            position_ids = torch.arange(cfg.seq_len, device='cuda')
            return {"hidden_states": hidden_states,
                    "position_embeddings": position_embeddings,
                    "cache_position": cache_position,
                    "position_ids": position_ids,
                    }
        
        def grads():
            grad = torch.randn(cfg.batch_size, cfg.seq_len, cfg.hidden_size, device='cuda', dtype=torch.bfloat16, requires_grad=False)
            return grad

        model = MyModel(cfg)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module = patch_linear_module(module, dropout=0.0)
        model = model.cuda().bfloat16()
        #print(model)
        runner.run(sys.argv, name, cfg.batch_size, cfg.seq_len, model, inputs, False, grads)
