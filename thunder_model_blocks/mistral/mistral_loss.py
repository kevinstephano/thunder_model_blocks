import sys
import torch

from thunder_model_blocks.utils import runner
from thunder_model_blocks.mistral import mistral_config
from transformers.models.mistral.modeling_mistral import MistralPreTrainedModel

class MyModel(MistralPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, logits: torch.Tensor, labels: torch.LongTensor):
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)
        return (loss,)

if __name__ == "__main__":
    cfg = mistral_config.config()

    def inputs(dtype, batch_size=cfg.batch_size, seq_len=cfg.seq_len, packed_seq_fn=None):
        args = {
            "logits": torch.randn(batch_size, seq_len, cfg.vocab_size, device='cuda', dtype=dtype, requires_grad=True),
            "labels": torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device='cuda', requires_grad=False),
        }
        return args

    runner.run(sys.argv, cfg.name_or_path, cfg, MyModel, inputs, module_has_loss=True)
