import torch
from torch import nn
import sys
from thunder_model_blocks.utils import runner

from transformers.models.llama import LlamaForCausalLM, LlamaConfig

LLAMA_3_2_1B_CFG = {
    "_name_or_path": "meta-llama/Llama-3.2-1B-Instruct",
    "architectures": ["LlamaForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 128000,
    "eos_token_id": 128001,
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 8192,
    "max_position_embeddings": 131072,
    "mlp_bias": False,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 16,
    "num_key_value_heads": 8,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
    "rope_theta": 500000.0,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.45.0.dev0",
    "use_cache": True,
    "vocab_size": 128256,
    "_commit_hash": "4e20de362430cd3b72f300e6b0f18e50e7166e08",
}

config = LlamaConfig(**LLAMA_3_2_1B_CFG)
config.batch_size = 1
config.seq_len = 6
config.num_hidden_layers = 1
config._attn_implementation = "sdpa"
configs = {}
configs[config.name_or_path] = config

class MyModel(torch.nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.model = LlamaForCausalLM(config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        cache_positions: torch.LongTensor,
        attention_mask: torch.LongTensor,
        use_cache: bool,
        ) :
        out = self.model(input_ids=input_ids,
                         cache_positions=cache_positions,
                         attention_mask=attention_mask,
                         use_cache=use_cache)
        return (out,)

if __name__ == "__main__":
    for name,cfg in configs.items():
        def inputs():
            args = dict(
                input_ids=torch.tensor([[128000, 791, 1401, 311, 2324, 374]], device="cuda"),
                cache_positions=torch.arange(6, device="cuda"),
                attention_mask=torch.ones(1, 6, dtype=torch.int64, device="cuda"),
                use_cache=True,
            )
            return args

        model = MyModel(cfg)
        model = model.cuda().bfloat16().requires_grad_(False).eval()
        #print(model)
        runner.run(sys.argv, name, cfg.batch_size, cfg.seq_len, model, inputs, False)
