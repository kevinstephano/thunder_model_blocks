import torch
import sys

from thunder_model_blocks.utils import runner
from thunder_model_blocks.hf_qwen2 import qwen2_config
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

class MyModel(torch.nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.model = Qwen2MLP(config)

    def forward(self, hidden_states: torch.Tensor):
        out = self.model(hidden_states)
        return (out,)

if __name__ == "__main__":
    config = qwen2_config.config()
    config.lora = True
    configs = {config.name_or_path: config}

    for name,cfg in configs.items():
        def inputs(dtype, batch_size=cfg.batch_size, seq_len=cfg.seq_len):
            hidden_states = torch.randn(batch_size, seq_len, cfg.hidden_size, device='cuda', dtype=dtype, requires_grad=True)
            return {"hidden_states": hidden_states}
        def grads(dtype, batch_size=cfg.batch_size, seq_len=cfg.seq_len):
            grad = torch.randn(batch_size, seq_len, cfg.hidden_size, device='cuda', dtype=dtype, requires_grad=False)
            return grad
        
        runner.run(sys.argv, name, cfg, MyModel, inputs, module_has_loss=False, grad_fn=grads)
