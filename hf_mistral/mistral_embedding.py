import sys
import torch

from thunder_model_blocks.utils import runner
from thunder_model_blocks.hf_mistral import mistral_config
from transformers.models.mistral import MistralPreTrainedModel

class MyModel(MistralPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input_ids: torch.LongTensor):
        inputs_embeds = self.embed_tokens(input_ids)
        return (inputs_embeds,)

if __name__ == "__main__":
    config = mistral_config.config()
    configs = {config.name_or_path: config}

    for name,cfg in configs.items():
        def inputs(dtype, batch_size=cfg.batch_size, seq_len=cfg.seq_len):
            input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device='cuda', requires_grad=False)
            return {"input_ids": input_ids}
        def grads(dtype, batch_size=cfg.batch_size, seq_len=cfg.seq_len):
            grad = torch.randn(batch_size, seq_len, cfg.hidden_size, device='cuda', dtype=dtype, requires_grad=False)
            return grad

        runner.run(sys.argv, name, cfg, MyModel, inputs, module_has_loss=False, grad_fn=grads)
