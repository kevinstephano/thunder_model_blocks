import json
from transformers.models.gemma2 import Gemma2Config

gemma2_cfg_str = r'''{
  "architectures": [
    "Gemma2ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "attn_logit_softcapping": 50.0,
  "bos_token_id": 2,
  "cache_implementation": "hybrid",
  "eos_token_id": 1,
  "final_logit_softcapping": 30.0,
  "head_dim": 256,
  "hidden_act": "gelu_pytorch_tanh",
  "hidden_activation": "gelu_pytorch_tanh",
  "hidden_size": 3584,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 8192,
  "model_type": "gemma2",
  "num_attention_heads": 16,
  "num_hidden_layers": 42,
  "num_key_value_heads": 8,
  "pad_token_id": 0,
  "query_pre_attn_scalar": 256,
  "rms_norm_eps": 1e-06,
  "rope_theta": 10000.0,
  "sliding_window": 4096,
  "sliding_window_size": 4096,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.42.0.dev0",
  "use_cache": true,
  "vocab_size": 256000
}
'''

def config():
    config = Gemma2Config.from_dict(json.loads(gemma2_cfg_str))
    config.batch_size = 1
    config.seq_len = 4096
    config._attn_implementation = "sdpa"
    return config
