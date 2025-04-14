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

class LitgptRope(torch.nn.Module):
    def __init__(self, config) -> None:
        from litgpt.model import apply_rope

        self.fused_apply_rotary_pos_emb_cached = None

        super().__init__()
        self.config = config
        self.apply_rope = apply_rope

    def forward(
        self,
        qkv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = qkv.shape  # batch size, sequence length

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size)
        qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)

        # maybe repeat k and v if for the non multi-head attention cases
        # training: flash attention requires it
        # inference: multi-query would require a full kv cache so avoid it to limit its memory usage
        if self.config.n_query_groups != self.config.n_head and self.config.n_query_groups != 1:
            k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
            v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)

        q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
        k = k.reshape(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
        v = v.reshape(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)

        q_roped = self.apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
        k_roped = self.apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
        k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)
        return q, k, v

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AutoModel on top of Thunder Based Model Examples')
    parser.add_argument('--model', default="meta-llama/Meta-Llama-3-8B-Instruct", type=str, help='HuggingFace model name/path.')
    args,extra_args = parser.parse_known_args(args=sys.argv[1:])
    assert args.model is not None, "The user did not provide a model!"
    sys_argv = [sys.argv[0]] + extra_args

    cfg = litgpt.Config.from_name(args.model)
    cfg.batch_size = 1
    cfg.seq_len = 4096
    cfg.name_or_path = args.model

    def inputs(dtype, batch_size=cfg.batch_size, seq_len=cfg.seq_len, packed_seq_fn=None):
        args = {
            "qkv": torch.randn(batch_size, seq_len, (cfg.n_head + 2 * cfg.n_query_groups) * cfg.head_size, device='cuda', dtype=dtype, requires_grad=True),
            "cos": torch.randn(seq_len, cfg.rope_n_elem, device='cuda', dtype=dtype, requires_grad=False),
            "sin": torch.randn(seq_len, cfg.rope_n_elem, device='cuda', dtype=dtype, requires_grad=False),
        }
        return args
    def grads(dtype, batch_size=cfg.batch_size, seq_len=cfg.seq_len):
        grad = torch.randn(batch_size, cfg.n_head, seq_len, cfg.head_size, device='cuda', dtype=dtype, requires_grad=False)
        return grad

    runner.run(sys_argv, cfg.name_or_path, cfg, LitgptRope, inputs, module_has_loss=False, grad_fn=grads)
