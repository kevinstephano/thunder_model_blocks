import sys
import torch

from thunder_model_blocks.utils import runner
from thunder_model_blocks.starcoder2 import starcoder2_config
from transformers.models.starcoder2 import Starcoder2ForCausalLM

class MyModel(torch.nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.model = Starcoder2ForCausalLM(config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        ) :
        out = self.model(input_ids=input_ids, labels=labels)
        assert out.loss is not None, "Loss is none?"
        return (out.loss,)

if __name__ == "__main__":
    config = starcoder2_config.config()
    config.lora = True
    configs = {config.name_or_path: config}

    for name,cfg in configs.items():
        def inputs(dtype, batch_size=cfg.batch_size, seq_len=cfg.seq_len):
            input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device='cuda', requires_grad=False)
            labels = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device='cuda', requires_grad=False)
            return {"input_ids": input_ids, "labels": labels}

        runner.run(sys.argv, name, cfg, MyModel, inputs, module_has_loss=True)
