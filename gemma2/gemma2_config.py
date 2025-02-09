from transformers import AutoConfig

def config():
    config = AutoConfig.from_pretrained("google/gemma-2-9b-it")
    config.batch_size = 1
    config.seq_len = 4096
    config._attn_implementation = "sdpa"
    return config
