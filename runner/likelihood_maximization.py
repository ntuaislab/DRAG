"""
likelihood_maximization.py

References
----------
.. [1] Singh, A., Chopra, A., Garza, E., Zhang, E., Vepakomma, P., Sharma, V., & Raskar, R.
       (2021). Disco: Dynamic and invariant sensitive channel obfuscation for deep neural
       networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
       Recognition (pp. 12125-12135).

       https://arxiv.org/pdf/2012.11025
"""

# pylint: disable=wrong-import-position
from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.utilities.types import (STEP_OUTPUT,
                                               OptimizerLRSchedulerConfig)
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torchmetrics.functional.image.tv import \
    _total_variation_update as total_variation  # pylint: disable=import-error
from torchvision.transforms import v2

from models import use_criterion
from models.distance import dino_image_similarity, patch_prior_distance
from models.likelihood_maximization import skip
from runner.data_reconstruction import ReconstructionMetric
from runner.utils import create_optimizer, create_scheduler, flatten_dictionary


class LitLikelihoodMaximization(L.LightningModule):
    """ Inversion task. """
    unnormalizer: nn.Module
    preprocessor: nn.Module

    mask: Tensor
    x_target: Tensor
    intermediate_repr: Tensor

    deep_image_prior: nn.Module

    def __init__(
        self,
        config: DictConfig,
        x_target: Tensor,
        intermediate_repr: Tensor,
        mask: Tensor,
        model: nn.Module | None = None,
        preprocessor: nn.Module | None = None,
        unnormalizer: nn.Module | None = None,
    ) -> None:
        super().__init__()

        assert model is not None
        assert unnormalizer is not None, "Unnormalizer is required."
        assert preprocessor is not None, "Preprocessor is required."

        # store hparams to `self._hparams`
        self.save_hyperparameters('config')

        n = intermediate_repr.shape[0]
        shape = tuple(OmegaConf.to_object([n] + config.parameters.shape))

        model_kwargs = OmegaConf.to_object(config.architecture_kwargs)

        self.config = config
        self.x = nn.Parameter(torch.randn(shape), requires_grad=False) # type: ignore
        self.register_buffer('x_target', x_target.clone())
        self.model = model.eval().requires_grad_(False)
        self.distance_fn = use_criterion(config.distance_fn)
        self.preprocessor = preprocessor
        self.unnormalizer = unnormalizer
        self.deep_image_prior = skip(**model_kwargs) # type: ignore
        self.register_buffer('intermediate_repr', intermediate_repr)
        self.register_buffer('mask', mask)

        self.metric = ReconstructionMetric(**dino_image_similarity())

    def unwrap(self) -> Tensor:
        return self.preprocessor(self.deep_image_prior(self.x)) # assume co-domain of f(x) is the domain of m(x)

    def configure_optimizers(self): # pylint: disable=missing-function-docstring
        optimizer_config = OmegaConf.to_container(self.config.optimizer, resolve=True)
        assert isinstance(optimizer_config, dict)
        optimizer = create_optimizer(
            optimizer_config.pop('_target_'),
            [
                { 'params': self.deep_image_prior.parameters() },
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
        # https://github.com/aidecentralized/InferenceBenchmark/blob/main/src/algos/input_model_optimization.py#L114
        x = self.x
        for step in self.config.noise_schedule:
            # Linear search until the current schedule step.
            if (self.global_step / self.config.max_steps) > step.milestone:
                continue

            # Perturb the input.
            x = self.x + self.x.detach().clone().normal_() * step.noise_std

            # Perturb the weights.
            for n in [w for w in self.deep_image_prior.parameters() if len(w) == 4]:
                n = n + n.detach().clone().normal_() * n.std() * step.parameter_noise_std

            break

        # Regularization.
        res = list(self.config.parameters.image_shape[-2:])
        h, w = res # pylint: disable=invalid-name
        tv_weight = self.config.regularization.total_variation
        patch_weight = self.config.regularization.patch_prior

        x = self.deep_image_prior(x) # assume co-domain of f(x) is [0, 1]
        x = v2.functional.resize(x, res, antialias=True)
        x = x.clamp(0, 1)
        x = self.preprocessor(x)
        hidden_state_distance = self.distance_fn(
            self.model(x).squeeze(0)[self.mask],
            self.intermediate_repr.squeeze(0)[self.mask],
        )

        tv_reg, _ = total_variation(x)
        tv_reg = tv_reg / (3 * h * w)

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

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        res = list(self.config.parameters.image_shape[-2:])

        x_pred = self.deep_image_prior(self.x) # assume co-domain of f(x) is [0, 1]
        x_pred = v2.functional.resize(x_pred, res, antialias=True)
        x_pred = x_pred.clamp(0, 1)

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

        with torch.no_grad():
            x = self.deep_image_prior(self.x)
            x = v2.functional.resize(x, res, antialias=True)
            # x = CLIP_IMG_UNNORMALIZE(x).clamp(0, 1)

        assert (writer := getattr(self.logger, 'experiment')) is not None
        writer.add_image('Reconstruction', x.squeeze(0), self.global_step)

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)

    def on_train_start(self):
        if self.logger is None:
            return

        score = {
            f'hp/{self.metric.keymap[k]}': torch.tensor(float('nan'))
                for k, _ in self.metric.items(True, False)
        }
        self.logger.log_hyperparams(flatten_dictionary(self.config), score)
