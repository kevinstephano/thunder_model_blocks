import torch
from torch import nn
import sys
from thunder_model_blocks.utils import runner
from thunder_model_blocks.utils.lora import patch_linear_module
#from nemo.collections.llm.peft.lora import patch_linear_module

from transformers import AutoConfig
from transformers.models.phi3 import Phi3ForCausalLM

config = AutoConfig.from_pretrained("microsoft/Phi-3.5-mini-instruct")

config.batch_size = 1
config.seq_len = 8192
config._attn_implementation = "sdpa"
configs = {}
configs[config.name_or_path] = config

class MyModel(torch.nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.model = Phi3ForCausalLM(config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        ) :
        out = self.model(input_ids=input_ids, labels=labels)
        assert out.loss is not None, "Loss is none?"
        return (out.loss,)

if __name__ == "__main__":
    for name,cfg in configs.items():
        def inputs():
            input_ids = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device='cuda', requires_grad=False)
            labels = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device='cuda', requires_grad=False)
            return {"input_ids": input_ids, "labels": labels}

        model = MyModel(cfg)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module = patch_linear_module(module, dropout=0.0)
        model = model.cuda().bfloat16()
        #print(model)
        runner.run(sys.argv, name, cfg.batch_size, cfg.seq_len, model, inputs, True)
