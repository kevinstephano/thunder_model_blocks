import torch
from torch import nn
import sys
from thunder_model_blocks.utils import runner
from thunder_model_blocks.utils.lora import patch_linear_module
#from nemo.collections.llm.peft.lora import patch_linear_module
 
from transformers import AutoModelForCausalLM
from transformers.models.phi3 import Phi3PreTrainedModel
 
config = AutoConfig.from_pretrained("microsoft/Phi-3.5-mini-instruct")
 
config.batch_size = 1
config.seq_len = 8192
configs = {}
configs[config.name_or_path] = config

class MyModel(Phi3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.padding_idx)

        # Initialize weights and apply final processing
        self.post_init()
 
    def forward(self, input_ids: torch.LongTensor = None):
        inputs_embeds = self.embed_tokens(input_ids)
        return (inputs_embeds,)
 
if __name__ == "__main__":
    for name,cfg in configs.items():
        def inputs():
            input_ids = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device='cuda', requires_grad=False)
            return {"input_ids": input_ids}
        def grads():
            grad = torch.randn(cfg.batch_size, cfg.seq_len, cfg.hidden_size, device='cuda', dtype=torch.bfloat16, requires_grad=False)
            return grad
 
        model = MyModel(cfg)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module = patch_linear_module(module, dropout=0.0)
        model = model.cuda().bfloat16()
        runner.run(sys.argv, name, cfg.batch_size, cfg.seq_len, model, inputs, False, grads)
