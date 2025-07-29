"""
defense.py
-----------

Frequently used defenses for privacy preserving machine learning.
"""

from pathlib import Path
from typing import Callable

import torch

from omegaconf import DictConfig, OmegaConf
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torch.distributions.laplace import Laplace

UNUSED = None

def _dropout(x: Tensor, p: float) -> Tensor:
    """ Zero out some digits of the input tensor with a prob.

    Arguments
    ---------
    x : Tensor
        Input tensor.

    p : float
        Probability of zeroing out the input tensor.

    Returns
    -------
    Tensor
        Zeroed out tensor.

    Reference
    ---------
    [1] He, Z., Zhang, T., & Lee, R. B. (2020). Attacking and protecting data privacy in
        edge–cloud collaborative inference systems. IEEE Internet of Things Journal, 8(12),
        9706-9716.
    """
    # f^{dropout}(x) = f(x) * m; m \in {0, 1}
    mask = (torch.rand_like(x) < p).to(torch.float32)
    return x * (1 - mask)


def _gaussian_noise(x: Tensor, p: float, mean : float = 0, std : float = 1) -> Tensor:
    """ Add Gaussian noise to the input tensor with a prob.

    Arguments
    ---------
    x : Tensor
        Input tensor.

    p : float
        Probability of adding noise to the input tensor.

    mean : float
        Mean of the Gaussian noise.

    std : float
        Standard deviation of the Gaussian noise.

    Returns
    -------
    Tensor
        Noised tensor.

    Reference
    ---------
    [1] He, Z., Zhang, T., & Lee, R. B. (2020). Attacking and protecting data privacy in
        edge–cloud collaborative inference systems. IEEE Internet of Things Journal, 8(12),
        9706-9716.
    """
    return x + torch.randn_like(x) * std + mean


def _laplace_noise(x: Tensor, p: float, loc: float = 0, scale: float = 1) -> Tensor:
    """ Add Laplacian noise to the input tensor with a prob.

    Arguments
    ---------
    x : Tensor
        Input tensor.

    p : float
        Probability of adding noise to the input tensor.

    loc : float
        Mean of the Laplacian noise.

    scale : float
        Scale of the Laplacian noise.

    Returns
    -------
    Tensor
        Noised tensor.

    Reference
    ---------
    [1] He, Z., Zhang, T., & Lee, R. B. (2020). Attacking and protecting data privacy in
        edge–cloud collaborative inference systems. IEEE Internet of Things Journal, 8(12),
        9706-9716.

    [2] Titcombe, T., Hall, A. J., Papadopoulos, P., & Romanini, D. (2021). Practical
        defences against model inversion attacks for split neural networks. arXiv preprint
        arXiv:2104.05743.
    """
    laplace = Laplace(loc, scale, validate_args=None)
    return x + laplace.sample(x.size())


def _shuffle(x: Tensor, p: float, output_permutation: bool = False) -> Tensor:
    """ Shuffle the input tensor with a prob.

    Arguments
    ---------
    x : (bs, n, d), Tensor
        Input tensor.

    p : float
        Probability of shuffling the input tensor.

    output_permutation : bool
        Whether to output permutation (h' := h[:, permutation]).
        Default is false.

    Returns
    -------
    Tensor
        Shuffled tensor in cloned form.
    """
    bs, n, _ = x.size()
    assert bs == 1, "Not support batch size > 1"

    N = n - 1  # exclude CLS token
    x = x.clone()
    device = x.device

    # CLS token is not involved into the permutation operation:
    # such that the shuffling operation does not affect the CLS pooling operation.
    rest_tokens = x[:, 1:, :]  # shape (bs, N, d)

    # Determine the number of tokens to permute based on the ratio
    if (num_permute := int(p * N)) == 0:
        return x

    # Randomly select a subset of the tokens to permute
    permute_indices = torch.randperm(N, device=device)[:num_permute]
    permute = torch.randperm(num_permute, device=device)

    rest_tokens[:, permute_indices] = rest_tokens[:, permute_indices[permute]]

    if output_permutation is True:
        src, dst = (permute_indices + 1), (permute_indices + 1)[permute]

        indices = torch.arange(n, device=device)
        indices[src] = indices[dst]

        return x, indices

    return x


def _drop_token(x: Tensor, p: float, output_mask: bool = False) -> Tensor:
    """ Shuffle the input tensor with a prob.

    Arguments
    ---------
    x : (bs, n, d), Tensor
        Input tensor.

    p : float
        Probability of shuffling the input tensor.

    Returns
    -------
    Tensor
        Shuffled tensor in cloned form.
    """
    bs, n, _ = x.size()
    assert bs == 1, "Not support batch size > 1"

    N = n - 1  # exclude CLS token
    x = x.clone()

    # CLS token is not involved into the dropout operation:
    # such that the dropout operation does not affect the CLS pooling operation.
    cls_token = x[:, :1, :]  # shape (bs, 1, d)
    rest_tokens = x[:, 1:, :]  # shape (bs, N, d)

    # Determine the number of tokens to permute based on the ratio
    if (num_drop := int(p * N)) == 0:
        return x

    # Randomly select a subset of the tokens to drop
    keep_mask = torch.randperm(N) >= num_drop
    x = torch.cat((cls_token, rest_tokens[:, keep_mask]), dim=1)

    if output_mask:
        mask = torch.ones(n, dtype=torch.bool, device=x.device)
        mask[1:][~keep_mask] = False
        return x, mask

    return x


def _reorder_tokens(
    h: Tensor,
    predictor: str,
    predictor_fn: Callable | None = None,
    output_permutation: bool = False,
    N: int | None = None,
) -> Tensor:
    """ Reorder the intermediate tensors with a cost function.

    Arguments
    ---------
    h : (bs, n, d), Tensor
        Input tensor.

    predictor : str
        Path to the predictor model.

    predictor_fn : Callable
        Predictor function.

    output_permutation : bool
        Whether to output permutation (h := h'[:, permutation]).
        Default is false.

    N : int
        Number of tokens in expected.

    Returns
    -------
    Tuple[Tensor, Tensor]
        Shuffled tensor and valid mask.

    Tensor
        Permutation indices.
    """
    bs, n, d = h.size()
    assert bs == 1, "Not support batch size > 1"

    N = N or n

    device = h.device

    if predictor_fn is None:
        ckpt = torch.load(Path(predictor) / 'last.ckpt', map_location='cpu')['state_dict']
        ckpt = {
            k.replace('position_predictor.', ''): v
                for k, v in ckpt.items()
                    if k.startswith('position_predictor.')
        }

        predictor_fn: nn.Module = nn.Sequential(
            nn.Linear(in_features=d, out_features=d),
            nn.ReLU(),
            nn.Linear(in_features=d, out_features=N)
        )
        predictor_fn.load_state_dict(ckpt)
        predictor_fn.eval().to(h.device)

    # CLS token is not involved into the permutation operation:
    cls_token = h[:, :1, :]  # shape (bs, 1, d)
    rest_tokens = h[:, 1:, :]  # shape (bs, n, d)
    reordered_tokens = torch.zeros((bs, N, d), device=device) # The dropped tokens are filled with zeros

    # Predict the position of the patch tokens
    logits: Tensor = predictor_fn(rest_tokens)
    logprobs = logits[:, :, 1:].softmax(dim=-1).squeeze(0).log()
    logprobs = torch.where(logprobs.isneginf(), -1e8, logprobs)

    src, dst = linear_sum_assignment(logprobs.cpu().numpy(), maximize=True)

    reordered_tokens[:, dst + 1] = rest_tokens[:, src]
    reordered_tokens[:, 0] = cls_token

    mask = torch.zeros(N, dtype=torch.bool, device=device)
    mask[dst + 1] = True
    mask[0] = True

    if output_permutation:
        indices = torch.arange(N, device=device)
        indices[dst + 1] = indices[src + 1]
        indices[0] = 0
        indices[~mask] = -1

        return (reordered_tokens, mask), indices

    return (reordered_tokens, mask)


def _channel_pruning(
    h: Tensor,
    pruner: Path,
    pruner_fn: Callable | None = None,
    p: float = UNUSED, # type: ignore
) -> Tensor:
    assert pruner_fn is None, "Not support custom pruner function."
    assert h.size(0) == 1, "Not support batch size > 1"

    if h.dim() not in (3, 4):
        raise ValueError(f"Expected input tensor to have shape (bs, n, d) or (bs, c, h, w), found {h.size()}.")

    if isinstance(pruner, str):
        pruner = Path(pruner)

    with open(pruner / '.hydra/config.yaml') as f:
        cfg = OmegaConf.load(f)
        assert isinstance(cfg, DictConfig)

    from ..disco import _normalize_score, load_pruner # Workaround: circular import
    pruner_fn = load_pruner(cfg, pruner / 'last.ckpt')
    pruner_fn.eval().to(h.device)

    match h.dim():
        case 3:
            s: Tensor = pruner_fn(h)
            s = _normalize_score(s, cfg.disco.pruning_ratio, cfg.disco.temperature)
            s = s[:, None, :]
            h = s * h

        case 4:
            s: Tensor = pruner_fn(h)
            s = _normalize_score(s, cfg.disco.pruning_ratio, cfg.disco.temperature)
            s = s[:, :, None, None]
            h = s * h

    return h


def apply_defense(x: Tensor, name: str | None, **kwargs) -> Tensor:
    """ Apply defense to the input tensor.

    Arguments
    ---------
    x : Tensor
        Input tensor. Shape: (N, C, H, W).

    name : str
        Defense name.

    kwargs : Dict
        Defense arguments.

    Returns
    -------
    Tensor
        Defensed tensor.
    """
    if name is None:
        return x

    if name in ('zero_out', 'dropout', 'DropoutDefense'):
        return _dropout(x, **kwargs)

    if name == 'gaussian_noise':
        return _gaussian_noise(x, **kwargs)

    if name == 'laplace_noise':
        return _laplace_noise(x, **kwargs)

    if name == 'shuffle':
        return _shuffle(x, **kwargs)

    if name == 'drop_token':
        return _drop_token(x, **kwargs)

    if name == 'channel_pruning':
        return _channel_pruning(x, **kwargs)

    raise ValueError(f"Unknown defense: {name}.")


def apply_adaptive_attack(intermediate_repr: Tensor, name: str | None, **kwargs) -> Tensor:
    """ Apply adaptive attack to the intermediate tensor.

    Arguments
    ---------
    intermediate_repr : Tensor
        Intermediate representation. Shape: (N, bs, D).

    name : str
        Attack name.

    kwargs : Dict
        Attack arguments.

    Returns
    -------
    Tensor
        Processed intermediate representation.
    """
    if name is None:
        return intermediate_repr

    if name == 'reorder':
        return _reorder_tokens(intermediate_repr, **kwargs)

    raise ValueError(f"Unknown adaptive attack: {name}.")
