import sys
import torch
from typing import Optional

from thunder_model_blocks.utils import runner
from thunder_model_blocks.deepseek_r1 import deepseek_r1_config
from thunder_model_blocks.deepseek_r1.modeling_deepseek import DeepseekV3ForCausalLM

class MyModel(torch.nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.model = DeepseekV3ForCausalLM(config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor]=None,
        use_cache: bool=False
        ) :
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache)
        return (out,)

if __name__ == "__main__":
    cfg = deepseek_r1_config.config()

    def inputs(dtype, batch_size=cfg.batch_size, seq_len=cfg.seq_len, packed_seq_fn=None):
        args = {
            "input_ids": torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device="cuda"),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.int64, device="cuda"),
            "use_cache": True,
        }
        return args

    runner.run(sys.argv, cfg.name_or_path, cfg, MyModel, inputs, module_has_loss=False, grad_fn=None, inference=True)
