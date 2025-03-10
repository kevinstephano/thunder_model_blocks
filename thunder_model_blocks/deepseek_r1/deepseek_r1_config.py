import os
from transformers import AutoConfig
from thunder_model_blocks.utils.download import download

config_file = "configuration_deepseek.py"
model_file = "modeling_deepseek.py"
if not os.path.exists(config_file):
    download("https://huggingface.co/deepseek-ai/DeepSeek-R1/resolve/main/configuration_deepseek.py", "configuration_deepseek")
if not os.path.exists(model_file):
    download("https://huggingface.co/deepseek-ai/DeepSeek-R1/resolve/main/modeling_deepseek.py", "modeling_deepseek")

def config():
    config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True)
    config.batch_size = 1
    config.seq_len = 4096
    config._attn_implementation = "eager"
    return config
