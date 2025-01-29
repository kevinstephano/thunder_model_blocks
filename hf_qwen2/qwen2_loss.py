import torch
from torch import nn
import sys
from thunder_model_blocks.utils import runner
from thunder_model_blocks.utils.lora import patch_linear_module
#from nemo.collections.llm.peft.lora import patch_linear_module

from transformers import AutoConfig
from transformers.models.qwen2 import Qwen2PreTrainedModel

config = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

config.batch_size = 1
config.seq_len = 4096
config._attn_implementation = "sdpa"
configs = {}
configs[config.name_or_path] = config

class MyModel(Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, hidden_states: torch.Tensor, labels: torch.LongTensor):
        logits = self.lm_head(hidden_states)

        loss = None
        print("logits", logits.size(), logits.stride(), logits.dtype, logits.requires_grad)
        print("labels", labels.size(), labels.stride(), labels.dtype)

        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)
        return (loss,)

if __name__ == "__main__":
    for name,cfg in configs.items():
        def inputs():
            hidden_states = torch.randn(cfg.batch_size, cfg.seq_len, cfg.hidden_size, device='cuda', dtype=torch.bfloat16, requires_grad=True)
            labels = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device='cuda', requires_grad=False)
            return {"hidden_states": hidden_states, "labels": labels}

        model = MyModel(cfg)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module = patch_linear_module(module, dropout=0.0)
        model = model.cuda().bfloat16()
        #print(model)
        runner.run(sys.argv, name, cfg.batch_size, cfg.seq_len, model, inputs, True)
