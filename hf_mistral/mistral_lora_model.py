import json
import sys
import torch
from torch import nn
from thunder_model_blocks.utils import runner

from nemo.collections.llm.peft.lora import patch_linear_module
from transformers.models.mistral import MistralForCausalLM, MistralConfig

'''
# Example to download model and config
from transformers import AutoConfig, AutoModel
model = AutoModel.from_pretrained("mistralai/Mistral-Nemo-Base-2407")
config = AutoConfig.from_pretrained("mistralai/Mistral-Nemo-Base-2407")
'''

mistral_cfg_str = r'''{
  "_name_or_path": "mistralai/Mistral-Nemo-Base-2407",
  "architectures": [
    "MistralForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 5120,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 128000,
  "model_type": "mistral",
  "num_attention_heads": 32,
  "num_hidden_layers": 40,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-05,
  "rope_theta": 1000000.0,
  "sliding_window": null,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.43.3",
  "use_cache": true,
  "vocab_size": 131072
}
'''

config = MistralConfig.from_dict(json.loads(mistral_cfg_str))
config.batch_size = 1
config.seq_len = 4096
config._attn_implementation = "sdpa"
configs = {}
configs[config.name_or_path] = config

class MyModel(torch.nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.model = MistralForCausalLM(config)

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
