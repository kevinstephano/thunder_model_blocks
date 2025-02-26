import sys
import torch

from thunder_model_blocks.utils import runner
from thunder_model_blocks.qwen2 import qwen2_config
from transformers.models.qwen2.modeling_qwen2 import Qwen2PreTrainedModel

class MyModel(Qwen2PreTrainedModel):
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
    cfg = qwen2_config.config()

    def inputs(dtype, batch_size=cfg.batch_size, seq_len=cfg.seq_len):
        hidden_states = torch.randn(batch_size, seq_len, cfg.hidden_size, device='cuda', dtype=dtype, requires_grad=True)
        labels = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device='cuda', requires_grad=False)
        return {"hidden_states": hidden_states, "labels": labels}

    runner.run(sys.argv, cfg.name_or_path, cfg, MyModel, inputs, module_has_loss=True)
