import sys
import torch
from typing import Optional

from thunder_model_blocks.utils import runner
from thunder_model_blocks.deepseek_r1 import deepseek_r1_config
from thunder_model_blocks.deepseek_r1.modeling_deepseek import DeepseekV3MoE

class MyModel(torch.nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.model = DeepseekV3MoE(config)

    def forward(self, hidden_states: torch.Tensor) :
        out = self.model(hidden_states=hidden_states)
        return (out,)

if __name__ == "__main__":
    cfg = deepseek_r1_config.config()
    cfg.batch_size = 1
    cfg.seq_len = 4096

    def inputs(dtype, batch_size=cfg.batch_size, seq_len=cfg.seq_len):
        args = {
            "hidden_states": torch.randn(batch_size, seq_len, cfg.hidden_size, device='cuda', dtype=dtype, requires_grad=False),
        }
        return args

    runner.run(sys.argv, cfg.name_or_path, cfg, MyModel, inputs, module_has_loss=False, grad_fn=None, inference=True)
