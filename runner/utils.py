"""
utils.py

Utility functions for the runner.
"""

import math
import os
import random
import warnings
from pathlib import Path
from typing import Any, Dict, overload

import gpustat
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch import Tensor, optim
from torch.optim import lr_scheduler
from torch.optim.optimizer import ParamsT


def clip_norm_(x: Tensor, max_norm) -> Tensor:
    """ Clip the tensor in-place by limiting the maximum norm.

    Arguments
    ---------
    x : Tensor
        Input tensor.

    max_norm : float
        Maximum norm.

    Returns
    -------
    Tensor
        Clipped tensor.
    """
    norm = torch.norm(x, p=2)
    if norm > max_norm:
        x.mul_(max_norm / norm)
    return norm


@overload
def flatten_dictionary(d: Dict, parent_key='', sep='.') -> Dict[str, Any]:
    ...

@overload
def flatten_dictionary(d: DictConfig, parent_key='', sep='.') -> Dict[str, Any]:
    ...

def flatten_dictionary(d: Dict | DictConfig, parent_key='', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten a nested dictionary.

    Arguments
    ---------
    d : dict | DictConfig
        Dictionary to be flattened.

    parent_key : str
        Parent key if the dictionary is nested.

    sep: str
        Separator for keys in the flattened dictionary.

    Returns
    -------
    Flattened dictionary.
    """
    if isinstance(d, DictConfig):
        d = OmegaConf.to_object(d)

    items = {}
    for k, v in d.items():
        assert isinstance(k, str)

        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items |= flatten_dictionary(v, new_key, sep=sep)
        elif isinstance(v, (list, tuple)):
            items[new_key] = str(v)
        else:
            items[new_key] = v

    return items


def parse_torch_dtype(dtype: str) -> torch.dtype:
    """
    Parse the torch dtype.

    Arguments
    ---------
    dtype : str
        Data type.

    Returns
    -------
    torch.dtype
        Torch data type.
    """
    return {
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
        'bfloat16': torch.bfloat16,
    }[dtype]


def save_tensor(x: Tensor, filename: str): # pylint: disable=invalid-name
    """
    Save the tensor.

    Arguments
    ---------
    x : Tensor
        a tensor.

    filename : str
        Filename.
    """
    x = x.detach().cpu()
    torch.save(x, filename)


@overload
def save_image(x: Tensor, filename: str | Path): # pylint: disable=invalid-name
    ...

@overload
def save_image(x: Image.Image, filename: str | Path): # pylint: disable=invalid-name
    ...

def save_image(x: Tensor | Image.Image, filename: str | Path): # pylint: disable=invalid-name
    """
    Save the image.

    Arguments
    ---------
    x : Tensor | Image.Image
        Image tensor. Should be in the dynamic range [0, 1].
        Shape: (1, C, H, W).

    filename : str
        Filename.
    """
    if isinstance(x, Tensor):
        assert 0 <= x.min() and x.max() <= 1, f'Invalid range: [{x.min()}, {x.max()}].'

        x = x.detach().cpu().squeeze(0)
        if x.size(0) == 1:
            x = x.expand(3, -1, -1)
        x = (x.permute(1, 2, 0) * 255).byte()  # Scale to [0, 255]
        x = x.numpy()
        x = Image.fromarray(x)

    if isinstance(filename, Path):
        filename = str(filename)

    x.save(f'{filename}.png', 'png')


def same_seeds(seed: int | None = None):
    """ Set random seed for reproducibility. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # RuntimeError: upsample_bilinear2d_aa_backward_out_cuda does not have a deterministic
    # implementation, but you set 'torch.use_deterministic_algorithms(True)'. You can turn
    # off determinism just for this operation, or you can use the 'warn_only=True' option,
    # if that's acceptable for your application. You can also file an issue at
    # https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic
    # support for this operation.
    #
    # torch.use_deterministic_algorithms(True)


def create_optimizer(
    name: str,
    param: ParamsT,
    **kwargs
) -> optim.Optimizer:
    """ Create optimizer.

    Details: https://pytorch.org/docs/stable/optim.html
    """
    if (ctor := getattr(optim, name, None)) is None:
        raise ValueError(f"Optimizer '{name}' not found.")

    return ctor(param, **kwargs)


def create_scheduler(
    name: str | None,
    optimizer: optim.Optimizer,
    **kwargs
) -> lr_scheduler.LRScheduler:
    """ Create learning rate scheduler.

    Details: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    """
    match name:
        case None:
            return lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=0)
        case 'CosineAnnealingLRWarmup':
            return CosineAnnealingLRWarmup(optimizer, **kwargs)
        case _ if (ctor := getattr(lr_scheduler, name, None)) is not None:
            return ctor(optimizer, **kwargs)
        case _:
            raise ValueError(f"Scheduler '{name}' not found.")

class CosineAnnealingLRWarmup(lr_scheduler.CosineAnnealingLR):
    def __init__(self, optimizer, warmup_iter, T_max, eta_min=0, last_epoch=-1):
        """
        Arguments
        ---------
        optimizer : torch.optim.Optimizer
            Target optimizer to schedule learning rate.

        warmup_iter : int
            Number of iterations for warmup.

        T_max : int
            Maximum number of iterations.

        eta_min : float, default=0
            Minimum learning rate.

        last_epoch : int, default=-1
            The index of last epoch.
        """
        # pylint: disable=too-many-arguments
        self.warmup_iter = warmup_iter
        super().__init__(optimizer, T_max, eta_min, last_epoch)

    def get_lr(self):
        # pylint: disable=line-too-long
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]

        if self.last_epoch < self.warmup_iter:
            return [
                base_lr * self.last_epoch / self.warmup_iter
                    for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) * (1.0 + math.cos((self.last_epoch - self.warmup_iter) / (self.T_max - self.warmup_iter) * math.pi))
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
        ]

    def _get_closed_form_lr(self):
        # pylint: disable=line-too-long
        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
                for base_lr in self.base_lrs
        ]


def default_num_workers() -> int:
    """ Return the number of workers per GPU based on AISLab's cluster usage. """
    if (n_gpus := len(gpustat.GPUStatCollection.new_query().gpus)) == 0:
        raise RuntimeError("No GPUs found on the host.")

    if (n_cores := os.cpu_count()) is None:
        raise RuntimeError("Unable to determine the number of CPU cores.")

    return int(n_cores * torch.cuda.device_count() / n_gpus)
