import json
from transformers.models.mistral import MistralConfig

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

def config():
    config = MistralConfig.from_dict(json.loads(mistral_cfg_str))
    config.batch_size = 1
    config.seq_len = 4096
    return config
