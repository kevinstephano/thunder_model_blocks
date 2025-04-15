import argparse
import sys
import torch
from typing import Optional

from transformers import AutoConfig, AutoModelForCausalLM
from thunder_model_blocks.utils import runner

class MyModel(torch.nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_mask: Optional[torch.Tensor]=None,
        position_ids: Optional[torch.Tensor]=None,
        ) :
        out = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask, position_ids=position_ids)
        assert out.loss is not None, "Loss is none?"
        return (out.loss,)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AutoModel on top of Thunder Based Model Examples')
    parser.add_argument('--model', default=None, type=str, help='HuggingFace model name/path.')
    args,extra_args = parser.parse_known_args(args=sys.argv[1:])
    assert args.model is not None, "The user did not provide a model!"
    sys_argv = [sys.argv[0]] + extra_args

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    cfg.batch_size = 1
    cfg.seq_len = 128

    def inputs(dtype, batch_size=cfg.batch_size, seq_len=cfg.seq_len, packed_seq_fn=None):
        attention_mask = None
        position_ids = None
        if packed_seq_fn is not None:
            attention_mask, position_ids = packed_seq_fn(batch_size=batch_size, seq_len=seq_len)
        args = {
            "input_ids": torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device='cuda', requires_grad=False),
            "labels": torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device='cuda', requires_grad=False),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        return args

    runner.run(sys_argv, cfg.name_or_path, cfg, MyModel, inputs, module_has_loss=True)
