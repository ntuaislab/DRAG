"""
base.py
--------

Data reconstruction task.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Literal

import torch
import torchmetrics  # pylint: disable=import-error
from omegaconf import DictConfig
from torch import Tensor, nn
from torchmetrics.functional.regression.cosine_similarity import \
    _cosine_similarity_compute  # pylint: disable=import-error
from torchmetrics.image import \
    MultiScaleStructuralSimilarityIndexMeasure as \
    MSSSIM  # pylint: disable=import-error
from torchmetrics.image import \
    PeakSignalNoiseRatio as PSNR  # pylint: disable=import-error
from torchmetrics.image import \
    StructuralSimilarityIndexMeasure as SSIM  # pylint: disable=import-error
from torchmetrics.image.lpip import \
    LearnedPerceptualImagePatchSimilarity as \
    LPIPS  # pylint: disable=import-error
from torchmetrics.regression import MeanSquaredError

from models import create_model, split_model

from ..base import RunnerBase
from ..dataset import create_normalizer_and_unnormalizer
from ..utils import parse_torch_dtype
from .defense import _reorder_tokens, apply_defense


class ImageSimilarityMetric(torchmetrics.Metric):
    sum_similarity: Tensor
    total: Tensor

    def __init__(
        self,
        embedding_model: nn.Module | None = None,
        preprocessing: Callable | None = None,
        reduction: Literal["sum", "mean"] = 'mean',
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.add_state('sum_similarity', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

        if isinstance(embedding_model, nn.Module):
            embedding_model = embedding_model.eval().requires_grad_(False)

        self._embedding_model = embedding_model
        self._preprocessor = preprocessing
        self.reduction = reduction

    def update(self, preds: Tensor, target: Tensor) -> None:
        """ Update with a batch of predictions and targets.

        Arguments
        ---------
        preds : Tensor
            Predictions. Shape: (N, C, H, W).

        target : Tensor
            Targets. Shape: (N, C, H, W).
        """
        if preds.shape != target.shape:
            raise ValueError(f"Expected `preds` and `target` to have the same shape, "
                             f"got {preds.shape} and {target.shape}.")

        n, _, _, _ = target.size() # pylint: disable=invalid-name

        if self._embedding_model is not None:
            self.sum_similarity += _cosine_similarity_compute(
                self._embedding_model(self._preprocessor(preds)).squeeze(1),
                self._embedding_model(self._preprocessor(target)).squeeze(1),
                reduction='sum'
            )

        self.total += n # pylint: disable=no-member

    def compute(self) -> Tensor:
        # pylint: disable=missing-function-docstring
        match self.reduction:
            case 'mean':
                return self.sum_similarity.float() / self.total
            case 'sum':
                return self.sum_similarity.float()
            case _:
                raise ValueError(f"Unknown reduction: {self.reduction}.")


class ReconstructionMetric(torchmetrics.MetricCollection):
    """ Composite metric to measure the performance of the data reconstruction.

    References
    ----------
    .. [1] Best Way to Deal with Small Images with MS-SSIM.
           https://github.com/VainF/pytorch-msssim/issues/28#issuecomment-847162076
    """

    keymap = {
        'MeanSquaredError': 'ImgDist',
        'PeakSignalNoiseRatio': 'PSNR',
        'StructuralSimilarityIndexMeasure': 'SSIM',
        'MultiScaleStructuralSimilarityIndexMeasure': 'MS-SSIM',
        'LearnedPerceptualImagePatchSimilarity': 'LPIPS',
        'ImageSimilarityMetric': 'Similarity',
    }

    def __init__(
        self,
        embedding_model: nn.Module | None = None,
        preprocessing: Callable | None = None,
        **kwargs
    ) -> None:
        super().__init__([
            MeanSquaredError(),
            PSNR(data_range=1.0, reduction='elementwise_mean'),
            SSIM(data_range=1.0, reduction='elementwise_mean'),
            MSSSIM(data_range=1.0, reduction='elementwise_mean'),
            LPIPS(net_type='alex', reduction='mean'),
            ImageSimilarityMetric(
                embedding_model, preprocessing, reduction='mean'
            ),
        ])
        self.eval().requires_grad_(False)

    def update(self, preds: Tensor, target: Tensor) -> None:
        # pylint: disable=missing-function-docstring
        preds, target = preds.contiguous(), target.contiguous()
        super().update(preds, target)

    def compute(self) -> Dict[str, Tensor]:
        # pylint: disable=missing-function-docstring
        # Legacy: map the metric names to the desired names
        metric = super().compute()
        return {self.keymap[k]: v for k, v in metric.items()}

    def forward(self, preds: Tensor, target: Tensor) -> Dict[str, Any]:
        # pylint: disable=missing-function-docstring
        metric = super().forward(preds, target)
        return {self.keymap[k]: v for k, v in metric.items()}


class DataReconstructionBase(RunnerBase):
    """ Data reconstruction task under split inference setting.

    Threat model:
    - The adversary has black-box or white-box access to the victim model.
    - The adversary has the embedding z of the private image x.
    - The adversary has white-box access to the bottom part of the victim model.
    - Auxiliary information.

    Objective:
    - Reconstruct the private image x from the embedding z.
    """
    client_model: nn.Module
    client_preprocessor: nn.Module
    client_unnormalizer: nn.Module

    def _prepare_model(self, **kwargs) -> None:
        cfg = self.configs
        name = cfg.model.name
        dtype = parse_torch_dtype(cfg.model.torch_dtype)
        checkpoint = cfg.model.checkpoint
        split_points = cfg.model.split_points

        assert isinstance(name, str)
        assert isinstance(split_points, str)

        from ..pretrained import from_pretrained  # Workaround: circular import
        if checkpoint is not None and (Path(checkpoint)).exists():
            client, _ = from_pretrained(Path(checkpoint)) # type: ignore
            self.logger.info('Loaded fine-tuned parameters from %s', checkpoint)
        else:
            model = create_model(name, checkpoint, torch_dtype=dtype)
            client, _ = split_model(model, name, split_points, False)

        self.client_model = client.eval().to(self.device)
        self.logger.info('Target model: %s (split: %s)', checkpoint, split_points)

        t, inv_t = create_normalizer_and_unnormalizer(name)
        self.client_preprocessor = t
        self.client_unnormalizer = inv_t

    def _forward(self, x_target: Tensor):
        cfg = self.configs

        defense = cfg.get('defense', DictConfig({ 'name': None, 'target': None, 'kwargs': {} }))
        self.logger.info('Defense: %s(%s) (%s)', defense.name, defense.target, defense.kwargs)

        adaptive_attack = cfg.get('adaptive_attack', DictConfig({ 'name': None, 'kwargs': {} }))
        self.logger.info('Adaptive attack: %s (%s)', adaptive_attack.name, adaptive_attack.kwargs)

        # Use pre-processing defense if specified.
        if defense.target == 'input':
            x = apply_defense(x_target, defense.name, **defense.kwargs)
        else:
            x = x_target

        intermediate_repr = self.client_model(x).detach()
        n_tokens = intermediate_repr.size(1)

        # Use post-processing defense if specified.
        if defense.target == 'intermediate':
            intermediate_repr = apply_defense(intermediate_repr, defense.name, **defense.kwargs)

        # Use adaptive attack if specified.
        if adaptive_attack.name == 'reorder':
            intermediate_repr, mask = _reorder_tokens(
                intermediate_repr,
                output_permutation=False,
                N=n_tokens,
                **adaptive_attack.kwargs,
            )
        else:
            mask = torch.ones(n_tokens, dtype=torch.bool)

        return intermediate_repr, mask
