import sys
import torch

from thunder_model_blocks.utils import runner
from thunder_model_blocks.phi3 import phi3_config
from transformers.models.phi3 import Phi3PreTrainedModel

class MyModel(Phi3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, hidden_states: torch.Tensor, labels: torch.LongTensor):
        logits = self.lm_head(hidden_states)
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)
        return (loss,)

if __name__ == "__main__":
    config = phi3_config.config()
    configs = {config.name_or_path: config}

    for name,cfg in configs.items():
        def inputs(dtype, batch_size=cfg.batch_size, seq_len=cfg.seq_len):
            hidden_states = torch.randn(batch_size, seq_len, cfg.hidden_size, device='cuda', dtype=dtype, requires_grad=True)
            labels = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device='cuda', requires_grad=False)
            return {"hidden_states": hidden_states, "labels": labels}

        runner.run(sys.argv, name, cfg, MyModel, inputs, module_has_loss=True)
