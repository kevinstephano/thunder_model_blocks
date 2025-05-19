import sys
import torch

from thunder_model_blocks.utils import runner
from thunder_model_blocks.phi3 import phi3_config
from transformers.models.phi3 import Phi3PreTrainedModel

class MyModel(Phi3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, logits: torch.Tensor, labels: torch.LongTensor):
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)
        return (loss,)

if __name__ == "__main__":
    cfg = phi3_config.config()

    def inputs(dtype, batch_size=cfg.batch_size, seq_len=cfg.seq_len, packed_seq_fn=None):
        args = {
            "logits": torch.randn(batch_size, seq_len, cfg.vocab_size, device='cuda', dtype=dtype, requires_grad=True),
            "labels": torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device='cuda', requires_grad=False),
        }
        return args

    runner.run(sys.argv, cfg.name_or_path, cfg, MyModel, inputs, module_has_loss=True)
