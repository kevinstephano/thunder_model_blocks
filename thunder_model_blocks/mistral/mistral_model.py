import sys
import torch

from thunder_model_blocks.utils import runner
from thunder_model_blocks.mistral import mistral_config
from transformers.models.mistral import MistralForCausalLM

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
    cfg = mistral_config.config()

    def inputs(dtype, batch_size=cfg.batch_size, seq_len=cfg.seq_len, packed_seq_fn=None):
        args = {
            "input_ids": torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device='cuda', requires_grad=False),
            "labels": torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device='cuda', requires_grad=False),
        }
        return args

    runner.run(sys.argv, cfg.name_or_path, cfg, MyModel, inputs, module_has_loss=True)
