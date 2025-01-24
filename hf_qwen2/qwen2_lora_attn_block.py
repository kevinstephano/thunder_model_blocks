import torch
from torch import nn
import sys
from thunder_model_blocks.utils import runner

from nemo.collections.llm.peft.lora import patch_linear_module
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

config = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

config.batch_size = 1
config.seq_len = 4096
config._attn_implementation = "sdpa"
configs = {}
configs[config.name_or_path] = config

"""
Hidden States torch.Size([1, 4096, 3584]) (14680064, 3584, 1) torch.bfloat16
Position Embs Cos torch.Size([1, 4096, 128]) (524288, 128, 1) torch.bfloat16
Position Embs Sin torch.Size([1, 4096, 128]) (524288, 128, 1) torch.bfloat16
attn mask? None
past_key_value DynamicCache()
cache_position tensor([   0,    1,    2,  ..., 4093, 4094, 4095], device='cuda:0')
Kwargs {'position_ids': tensor([[   0,    1,    2,  ..., 4093, 4094, 4095]], device='cuda:0'), 
        'output_attentions': False, 
        'use_cache': True}
"""

class MyModel(torch.nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.model = Qwen2Attention(config, 0)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        cache_position,
        position_ids,
        ) :
        kwargs = {"position_ids": position_ids, "output_attentions": False, "use_cache": True}
        out,_ = self.model(hidden_states=hidden_states,
                         position_embeddings=position_embeddings,
                         attention_mask=None,
                         past_key_value=DynamicCache(),
                         cache_position=cache_position,
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
