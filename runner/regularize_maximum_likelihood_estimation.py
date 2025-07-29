"""
regularize_maximum_likelihood_estimation.py

References
----------
.. [1] He, Z., Zhang, T., & Lee, R. B. (2019, December). Model inversion attacks
       against collaborative inference. In Proceedings of the 35th Annual Computer
       Security Applications Conference (pp. 148-162).
       https://par.nsf.gov/servlets/purl/10208164
"""

import lightning as L
import torch
from lightning.pytorch.utilities.types import (STEP_OUTPUT,
                                               OptimizerLRSchedulerConfig)
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torchmetrics import Metric, MetricCollection
from torchmetrics.functional.image.tv import \
    _total_variation_update as total_variation  # pylint: disable=import-error
from torchvision.transforms import v2
from torchvision.utils import make_grid

from models import use_criterion
from models.distance import dino_image_similarity, patch_prior_distance
from runner.data_reconstruction import ReconstructionMetric
from runner.utils import create_optimizer, create_scheduler, flatten_dictionary

RESCALE = v2.Normalize([0.5], [0.5])

class LitRegularizeMaximumLikelihoodEstimation(L.LightningModule):
    unnormalizer: nn.Module
    preprocessor: nn.Module

    mask: Tensor
    x_target: Tensor
    intermediate_repr: Tensor

    def __init__(
        self,
        config: DictConfig,
        x_target: Tensor,
        intermediate_repr: Tensor,
        mask: Tensor,
        model: nn.Module | None = None,
        preprocessor: nn.Module | None = None,
        unnormalizer: nn.Module | None = None,
        metric: Metric | MetricCollection | None = None,
    ) -> None:
        super().__init__()

        assert intermediate_repr is not None, "Intermediate representation is required."
        assert model is not None

        # store hparams to `self._hparams`
        self.save_hyperparameters('config')

        batch_size = intermediate_repr.shape[0]
        shape = OmegaConf.to_object([batch_size] + config.parameters.shape) # type: ignore
        assert isinstance(shape, list), "Expected shape to be a list."

        init_x = torch.zeros(shape)

        assert list(init_x.size()) == shape, f"Expected shape {shape}, but got {init_x.size()}"
        assert unnormalizer is not None, "Unnormalizer is required."
        assert preprocessor is not None, "Preprocessor is required."

        self.config = config
        self.x = nn.Parameter(init_x)
        self.register_buffer('x_target', x_target.clone())
        self.model = model.eval().requires_grad_(False)
        self.distance_fn = use_criterion(config.distance_fn)
        self.preprocessor = preprocessor
        self.unnormalizer = unnormalizer
        self.register_buffer('intermediate_repr', intermediate_repr)
        self.register_buffer('mask', mask)

        if metric is None:
            metric = ReconstructionMetric(**dino_image_similarity())

        self.metric = metric

    def unwrap(self) -> Tensor:
        """
        Unwraps the tensor data from the object.

        Returns
        -------
        Tensor
            The unwrapped tensor data.
        """
        return self.x.data

    def configure_optimizers(self): # pylint: disable=missing-function-docstring
        optimizer_config = OmegaConf.to_container(self.config.optimizer, resolve=True)
        assert isinstance(optimizer_config, dict)
        optimizer = create_optimizer(
            optimizer_config.pop('_target_'),
            [
                { 'params': self.x },
            ],
            **optimizer_config, # type: ignore
        )

        scheduler_config = OmegaConf.to_container(self.config.scheduler, resolve=True)
        assert isinstance(scheduler_config, dict)
        scheduler = create_scheduler(
            scheduler_config.pop('_target_'),
            optimizer,
            **scheduler_config, # type: ignore
        )

        return OptimizerLRSchedulerConfig(
            optimizer=optimizer,
            lr_scheduler={
                "scheduler": scheduler,
            },
        )

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT: # pylint: disable=unused-argument
        # Regularization.
        res = list(self.config.parameters.image_shape[-2:])
        h, w = res # pylint: disable=invalid-name
        tv_weight = self.config.regularization.total_variation
        patch_weight = self.config.regularization.patch_prior

        x = v2.functional.resize(self.x, res, antialias=True)
        hidden_state_distance = self.distance_fn(
            self.model(x).squeeze(0)[self.mask],
            self.intermediate_repr.squeeze(0)[self.mask],
        )

        tv_reg, _ = total_variation(self.unnormalizer(x)) # but, measure total variation on
                                                          # [0, 1] domain
        tv_reg = tv_reg.mean() / (3 * h * w)

        if patch_weight:
            patch_prior = patch_prior_distance(x, self.model.config.patch_size)
        else:
            patch_prior = torch.tensor(0.0)

        self.log('Training/Hidden State Distance', hidden_state_distance.item(), prog_bar=True)
        self.log('Training/Total Variation', tv_reg.item())
        self.log('Training/Patch Prior', patch_prior.item())

        return hidden_state_distance \
               + tv_weight * tv_reg \
               + patch_weight * patch_prior

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT: # pylint: disable=unused-argument, missing-function-docstring
        res = list(self.config.parameters.image_shape[-2:])

        x_pred = self.unnormalizer(v2.functional.resize(self.x, res, antialias=True)).clamp(0, 1)
        self.metric.update(x_pred, self.x_target)

    def on_validation_epoch_end(self): # pylint: disable=missing-function-docstring
        if self.logger is None:
            return

        res = list(self.config.parameters.image_shape[-2:])
        score = {
            f'hp/{k}': v.item() for k, v in self.metric.compute().items()
        }
        self.metric.reset()
        self.log_dict(score)

        # Check if x_pred is batched
        x_pred = self.unnormalizer(v2.functional.resize(self.x, res, antialias=True)).clamp(0, 1)
        if x_pred.ndim == 4:
            x_pred = make_grid(x_pred, nrow=x_pred.size(0))

        assert (writer := getattr(self.logger, 'experiment')) is not None
        writer.add_image('Reconstruction', x_pred, self.global_step)

    def on_before_optimizer_step(self, optimizer): # pylint: disable=unused-argument, missing-function-docstring
        x_target = self.x_target
        x, grad = self.x, self.x.grad
        size = list(x_target.shape[-2:])

        assert grad is not None, "Gradient is None. Please check the optimizer step."

        if x_target.ndim == 3:
            x_target = x_target.unsqueeze(0)

        if x_target.shape != x.shape:
            x = v2.functional.resize(x, size, antialias=True)
            grad = v2.functional.resize(grad, size, antialias=True)

        self.log('Training/Gradient Norm', grad.norm().item())

    def on_train_start(self): # pylint: disable=missing-function-docstring
        if self.logger is None:
            return

        score = {
            f'hp/{self.metric.keymap[k]}': torch.tensor(float('nan'))
                for k, _ in self.metric.items(True, False)
        }
        self.logger.log_hyperparams(flatten_dictionary(self.config), score)
