from collections import OrderedDict
import torch
from torch import nn
import sys
from thunder_model_blocks.utils import runner

from nemo.collections.llm.peft.lora import patch_linear_module
from transformers import AutoConfig
from transformers.models.phi3.modeling_phi3 import Phi3MLP

config = AutoConfig.from_pretrained("microsoft/Phi-3.5-mini-instruct")

config.batch_size = 1
config.seq_len = 8192
config._attn_implementation = "sdpa"
configs = {}
configs[config.name_or_path] = config

class MyModel(torch.nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.model = Phi3MLP(config)

    def forward(self, hidden_states: torch.Tensor):
        out = self.model(hidden_states=hidden_states)
        return (out,)

if __name__ == "__main__":
    for name,cfg in configs.items():
        def inputs():
            hidden_states = torch.randn(cfg.batch_size, cfg.seq_len, cfg.hidden_size, device='cuda', dtype=torch.bfloat16, requires_grad=True)
            return {"hidden_states": hidden_states}
        def grads():
            grad = torch.randn(cfg.batch_size, cfg.seq_len, cfg.hidden_size, device='cuda', dtype=torch.bfloat16, requires_grad=False)
            return grad

        model = MyModel(cfg)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module = patch_linear_module(module, dropout=0)
        model = model.cuda().bfloat16()
        #print(model)
        runner.run(sys.argv, name, cfg.batch_size, cfg.seq_len, model, inputs, False, grads)
