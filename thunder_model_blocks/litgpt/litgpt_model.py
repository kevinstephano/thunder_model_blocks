import argparse
import subprocess
import sys
import torch

from thunder_model_blocks.utils import runner

def install_litgpt():
    """
    Check if litgpt is installed, and if not, install it using pip.
    Returns True if litgpt is successfully installed or already present,
    False if installation fails.
    """
    try:
        import litgpt as gpt 
        return gpt
    except ImportError:
        print("Litgpt is not installed. Installing now...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", '--no-deps', "litgpt[all]"])
            print("litgpt has been successfully installed!")
            import litgpt as gpt
            return gpt
        except subprocess.CalledProcessError:
            print("Failed to install litgpt. Please try installing manually using:")
            print("pip install --no-deps litgpt[all]")
            return None

litgpt = install_litgpt()

class MyModel(torch.nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.model = litgpt.GPT(config)

    def forward(self, idx: torch.LongTensor) :
        out = self.model(idx=idx)
        return (out.sum(),)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AutoModel on top of Thunder Based Model Examples')
    parser.add_argument('--model', default=None, type=str, help='HuggingFace model name/path.')
    args,extra_args = parser.parse_known_args(args=sys.argv[1:])
    assert args.model is not None, "The user did not provide a model!"
    sys_argv = [sys.argv[0]] + extra_args

    cfg = litgpt.Config.from_name(args.model)
    cfg.batch_size = 1
    cfg.seq_len = 2048
    cfg.name_or_path = args.model

    def inputs(dtype, batch_size=cfg.batch_size, seq_len=cfg.seq_len, packed_seq_fn=None):
        args = {
            "idx": torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device='cuda', requires_grad=False),
        }
        return args

    runner.run(sys_argv, cfg.name_or_path, cfg, MyModel, inputs, module_has_loss=True)
