import sys
import torch

from thunder_model_blocks.utils import runner
from thunder_model_blocks.hf_llama import llama_config
from transformers.models.llama import LlamaForCausalLM

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
    config = llama_config.config()
    configs = {config.name_or_path: config}

    for name,cfg in configs.items():
        def inputs(dtype, batch_size=cfg.batch_size, seq_len=cfg.seq_len):
            args = dict(
                #input_ids=torch.tensor([[128000, 791, 1401, 311, 2324, 374]], device="cuda"),
                input_ids=torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device="cuda"),
                cache_positions=torch.arange(seq_len, device="cuda"),
                attention_mask=torch.ones(batch_size, seq_len, dtype=torch.int64, device="cuda"),
                use_cache=True,
            )
            return args

        runner.run(sys.argv, name, cfg, MyModel, inputs, module_has_loss=False, grad_fn=None, inference=True)
