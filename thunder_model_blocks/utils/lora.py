import math
import torch
import torch.nn.functional as F
from torch import nn

class LinearAdapter(nn.Linear):
    """
    Linear + LoRA, maintains ckpts structrue (i.e. Linear's weight/bias remain at the same FQN)

    The _init_wrapper and _forward methods provide the LoRA functionality. We want to be able to
    use those inside LinearAdapter but also for monkey-patching modules, without repeating the
    same code -> therefore those are decorated with @staticmethod.

    Args:
        orig_linear (nn.Module): the linear module to augment.
        dim (int): lora's dim in_features -> dim -> out_features.
        alpha (int): lora's scaling alpha.
        dropout (float): dropout prob (default: 0.0).
        dropout_position (str): where to apply dropout rel. to lora (choices= ['pre', 'post'], default=post)
        lora_A_init_method (str): init method for lora_A (choices= ['xavier', 'uniform'])
        lora_dtype (torch.dtype): weight's dtype, by default will use orig_linear's but if they
        are quantized weights (e.g. 4bit) needs to be specified explicitly.
    """

    def __init__(
        self,
        orig_linear,
        dim=8,
        alpha=32,
        dropout=0.0,
        dropout_position='post',
        lora_A_init_method='xavier',
        lora_dtype=None,
    ):
        assert isinstance(orig_linear, nn.Linear)
        super(LinearAdapter, self).__init__(
            in_features=orig_linear.in_features,
            out_features=orig_linear.out_features,
            bias=orig_linear.bias is not None,
            device=orig_linear.weight.device,
            dtype=orig_linear.weight.dtype,
        )
        # copy weights
        self.weight.data.copy_(orig_linear.weight.data)
        if orig_linear.bias is not None:
            self.bias.data.copy_(orig_linear.bias.data)
        # initialize the adapte
        LinearAdapter._init_adapter(
            self,
            dim=dim,
            alpha=alpha,
            dropout=dropout,
            dropout_position=dropout_position,
            lora_A_init_method=lora_A_init_method,
            lora_dtype=lora_dtype,
        )

    @staticmethod
    def _init_adapter(
        obj,
        dim=8,
        alpha=32,
        dropout=0.0,
        dropout_position='post',
        lora_A_init_method='xavier',
        lora_dtype=None,
    ):
        """Adds LoRA weights to obj. The obj is either a LinearAdapter or an nn.Module (when
        monkey-patching).

        Args:
            obj (LinearAdapter | nn.Module): input module to adapt.
            dim (int): lora's dim in_features -> dim -> out_features.
            alpha (int): lora's scaling alpha.
            dropout (float): dropout prob (default: 0.0).
            dropout_position (str): where to apply dropout rel. to lora (choices= ['pre', 'post'], default=post)
            lora_A_init_method (str): init method for lora_A (choices= ['xavier', 'uniform'])
            lora_dtype (torch.dtype): weight's dtype, by default will use orig_linear's but if they
            are quantized weights (e.g. 4bit) needs to be specified explicitly.
        """
        obj.dim = dim
        obj.scale = alpha / dim

        # Freezer
        device = obj.weight.device
        obj.weight.requires_grad = False
        if obj.bias is not None:
            obj.bias.requires_grad = False

        in_features = obj.in_features
        out_features = obj.out_features
        dtype = lora_dtype or obj.weight.dtype

        obj.lora_a = nn.Parameter(torch.zeros((in_features, dim), dtype=dtype, device=device))
        obj.lora_b = nn.Parameter(torch.zeros((dim, out_features), dtype=dtype, device=device))
        if lora_A_init_method == 'xavier':
            torch.nn.init.uniform_(obj.lora_a)
        else:
            nn.init.kaiming_uniform_(obj.lora_a, a=math.sqrt(5))

        obj.dropout = nn.Dropout(p=dropout)
        assert dropout_position in ['pre', 'post'], dropout_position
        obj.dropout_position = dropout_position

    @staticmethod
    def _forward(obj, x, fwd=None):
        # pylint: disable=C0115,C0116
        if fwd is not None:
            res = fwd(x)
        else:
            res = F.linear(x, obj.weight, obj.bias)
        if obj.dropout_position == 'pre':
            x = obj.dropout(x)
        lora_res = x @ obj.lora_a
        lora_res = lora_res @ obj.lora_b
        lora_res = lora_res * obj.scale
        if obj.dropout_position == 'post':
            lora_res = obj.dropout(lora_res)
        return res + lora_res

    def forward(self, input):
        return LinearAdapter._forward(self, input)


def patch_linear_module(
    orig_linear,
    dim=8,
    alpha=32,
    dropout=0.0,
    dropout_position='post',
    lora_A_init_method='xavier',
    lora_dtype=None,
):
    """Monkey-patches a nn.Linear (orig_linear param) to be a LinearAdapter, for all purposes
    think of this function as replacing a nn.Linear with a LinearAdapter defined above.

    The orig_linear might not contain valid weights, for example, the given orig_linear was
    initialized within a context-manager that uses a "meta" device. Therefore, we cannot copy
    the weight/bias from the orig_linear to the LinearAdapter, since those have not been allocated,

    To circumvent this scenario, LinearAdapter's additional functionality (_init_adapter, _forward)
    is based on static functions, so that we can use them for patching or when allocating a
    new LinearAdapter object.

    Args:
        orig_linear (nn.Linear): the module we add adapter to.
        dim (int, optional): Lora dim. Defaults to 8.
        alpha (int, optional): Lora alpha scale. Defaults to 32.
        dropout (float, optional): dropout prob. Defaults to 0.0.
        dropout_position (str, optional): location to apply dropout wrt lora.
            Defaults to 'post' (choices: 'pre', 'post').
        lora_A_init_method (str, optional): lora_a init method. Defaults to 'xavier'.
        lora_dtype (_type_, optional): Lora weights' dtype. By default will use orig_linear's dtype
        but orig_linear might use non-trainable dtype (e.g. 4bit), in which case the user must
        specify the dtype manually. Defaults to None.

    Returns:
        (nn.Module): the monkey-patched (nn.Linear + LoRA) nn.Module
    """

    assert isinstance(orig_linear, nn.Linear)

    LinearAdapter._init_adapter(orig_linear, dim, alpha, dropout, dropout_position, lora_A_init_method, lora_dtype)
    fwd = None
    # If the model uses quantized weights, we want to use orig_linear's forward
    if orig_linear.weight.dtype == torch.uint8:
        fwd = orig_linear.forward
    orig_linear.forward = lambda input: LinearAdapter._forward(orig_linear, input, fwd)
    return orig_linear

