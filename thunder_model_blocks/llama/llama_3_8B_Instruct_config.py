import json
from transformers.models.llama import LlamaConfig

llama_3_8B_Instruct_cfg_str = r'''{
  "_name_or_path": "meta-llama/Meta-Llama-3-8B-Instruct",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "_name_or_path": "meta-llama/Meta-Llama-3-8B-Instruct",
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "eos_token_id": 128009,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 8192,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 500000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.40.0.dev0",
  "use_cache": true,
  "vocab_size": 128256
}
'''

def config():
    config = LlamaConfig.from_dict(json.loads(llama_3_8B_Instruct_cfg_str))
    config.batch_size = 1
    config.seq_len = 4096
    config._attn_implementation = "sdpa"
    return config
