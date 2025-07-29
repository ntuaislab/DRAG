import torch
from torch import Tensor


def f(x: Tensor, clone: bool = False) -> Tensor:
    return x.clone() if clone else x


@torch.no_grad()
def get_step(
    x: Tensor,
    optimizer: torch.optim.Optimizer,
    clone: bool = False,
) -> Tensor:
    """ Get the step for a parameter `x` in the optimizer `optimizer`.

    Reusing state information from the PyTorch classes.

    Parameters
    ----------
    x : Tensor
        The parameter tensor.

    optimizer : torch.optim.Optimizer
        The optimizer.

    clone : bool
        Select True to backup a tensor to avoid further in-place modification.
        Default is `False`.

    Returns
    -------
    Tensor
        Current step for the parameter `x` in the optimizer `optimizer`.
    """

    match type(optimizer):
        case torch.optim.SGD:
            if optimizer.defaults['momentum'] == 0:
                return f(x.grad, clone=clone) # type: ignore

            return f(optimizer.state[x]['momentum_buffer'], clone=clone)

        case torch.optim.Adam:
            (beta1, beta2), eps = optimizer.defaults['betas'], optimizer.defaults['eps']
            t = optimizer.state[x]['step']

            bias_correction1 = 1 - beta1 ** t
            bias_correction2 = 1 - beta2 ** t
            bias_correction2_sqrt = bias_correction2 ** 0.5

            exp_avg, exp_avg_sq = optimizer.state[x]['exp_avg'], optimizer.state[x]['exp_avg_sq']
            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
            return (exp_avg / bias_correction1) / denom

        case _:
            raise ValueError(f'Unsupported optimizer: {type(optimizer)}')
