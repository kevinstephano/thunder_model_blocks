from transformers import AutoConfig

def config():
    config = AutoConfig.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    config.batch_size = 1
    config.seq_len = 8192
    return config
