"""
detokenization.py
-----------------

Impl of detokenizer, decoder part of ViTMAE.
"""

from pathlib import Path
from typing import Callable

import lightning as L
import torch
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.utilities.types import (STEP_OUTPUT,
                                               OptimizerLRSchedulerConfig)
from omegaconf import DictConfig
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torchmetrics.classification import MulticlassAccuracy
from torchvision.transforms import functional as F
from torchvision.utils import make_grid
from tqdm import tqdm

from models.detokenizer import Detokenizer, get_tokenization_config
from models.distance import dino_image_similarity

from .base import VAL, Metric
from .data_reconstruction import (LearningBasedDRA, ReconstructionMetric,
                                  apply_adaptive_attack, apply_defense)
from .utils import create_optimizer, create_scheduler, flatten_dictionary


class DetokenizationRunner(LearningBasedDRA):
    """ Reconstruct data via reconstruct patch pixel. """
    position_predictor: nn.Module
    reconstructor: Detokenizer

    _image_size: int
    _patch_size: int

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.metric = ReconstructionMetric()

    def _prepare_model(self, **kwargs):
        super()._prepare_model(**kwargs)

        scaling_factor = self.configs.scaling_factor

        patch_size, image_size, embed_dim = get_tokenization_config(self.client_model)

        assert patch_size % scaling_factor == 0, \
            "Patch size must be divisible by scaling factor."

        patch_size //= scaling_factor
        image_size //= scaling_factor
        num_patch = (image_size // patch_size) ** 2

        position_predictor = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=embed_dim),
            nn.ReLU(),
            nn.Linear(in_features=embed_dim, out_features=num_patch + 1)
        )
        reconstructor = Detokenizer(
            embed_dim=embed_dim,
            patch_size=patch_size,
            image_size=image_size,
            **self.configs.reconstructor,
        )

        self._image_size = image_size
        self._patch_size = patch_size

        self.position_predictor = position_predictor.to(self.device)
        self.reconstructor = reconstructor.to(self.device)
        self.client_model = self.client_model.to(self.device)

    def load_checkpoint(self, dirname: str | Path):
        dirname = Path(dirname) / 'last.ckpt'
        state_dict = torch.load(dirname, map_location=self.device)['state_dict']

        self.reconstructor.load_state_dict({
            k.replace('reconstructor.', ''): v
                for k, v in state_dict.items()
                    if k.startswith('reconstructor.')
        })

        self.position_predictor.load_state_dict({
            k.replace('position_predictor.', ''): v
                for k, v in state_dict.items()
                    if k.startswith('position_predictor.')
        })

        return self

    @torch.no_grad()
    def run(self):
        self._prepare_dataset()
        self._prepare_model()

        Path('outputs').mkdir(exist_ok=True, parents=True)

        defense = self.configs.get('defense', DictConfig({ 'name': None, 'target': None, 'kwargs': {} }))
        self.logger.info('Defense: %s(%s) (%s)', defense.name, defense.target, defense.kwargs)

        adaptive_attack = self.configs.get('adaptive_attack', DictConfig({ 'name': None, 'kwargs': {} }))
        self.logger.info('Adaptive attack: %s (%s)', adaptive_attack.name, adaptive_attack.kwargs)

        for i, (img, *_) in tqdm(enumerate(self.dataloader[VAL])):
            img = img.to(self.device)

            # encode the image to intermediate representations
            intermediate_repr = self.client_model(img)

            # Use post-processing defense if specified.
            if defense.target == 'intermediate':
                intermediate_repr = apply_defense(intermediate_repr, defense.name, **defense.kwargs)

            # Use adaptive attack if specified.
            if adaptive_attack.name is not None:
                adaptive_attack_kwargs = { 'predictor': None, 'predictor_fn': self.position_predictor }
                intermediate_repr = apply_adaptive_attack(intermediate_repr, adaptive_attack.name, **adaptive_attack_kwargs)

            # generate the image based on the image embeddings
            img_pred = self.generate(intermediate_repr)
            img_pred = self.client_unnormalizer(img_pred).clamp(0, 1)

            # make_grid and save the image
            img_grid = make_grid(img_pred, nrow=8)
            F.to_pil_image(img_grid).save(f'output/{self.configs.model.checkpoint.replace("/", "-")}-{self.configs.model.split_points}-{i}.png')


    @torch.no_grad()
    def generate(
        self,
        intermediate_repr: Tensor,
        **kwargs
    ) -> Tensor:
        """ Generate the image from the model.

        Returns
        -------
        Tensor
            Generated image.
        """
        _mode = self.reconstructor.training

        # self.position_predictor.eval()
        self.reconstructor.eval()

        # predict the patch pixels given the image embeddings (drop CLS token)
        img_pred = self.reconstructor(intermediate_repr)

        self.reconstructor.train(_mode)
        # self.position_predictor.train(_mode)
        return img_pred

    @torch.no_grad()
    def evaluate(self, **kwargs) -> Metric:
        """ Evaluate the model.

        Returns
        -------
        Metric
            Evaluation metric.
        """
        self.metric.reset()

        num_patch_per_row = self._image_size // self._patch_size
        x = torch.arange(0, num_patch_per_row)
        y = torch.arange(0, num_patch_per_row)
        x, y = torch.meshgrid(x, y, indexing='ij')
        coordinates = torch.stack((x, y), dim=-1).unsqueeze(0).to(self.rank)

        accuracy = MulticlassAccuracy(
            1 + (self._image_size // self._patch_size) ** 2
        ).to(self.device)

        # defense = self.configs.get('defense', DictConfig({ 'name': None, 'target': None, 'kwargs': {} }))
        # self.logger.info('Defense: %s(%s) (%s)', defense.name, defense.target, defense.kwargs)

        # adaptive_attack = self.configs.get('adaptive_attack', DictConfig({ 'name': None, 'kwargs': {} }))
        # self.logger.info('Adaptive attack: %s (%s)', adaptive_attack.name, adaptive_attack.kwargs)

        position_err = 0
        device = self.device
        dataloader = self.dataloader[VAL]
        pbar = tqdm(dataloader, ncols=0, desc='Evaluating...', disable=self.rank != 0)
        for img, *_ in pbar:
            img = img.to(device)

            # encode the image to intermediate representations
            intermediate_repr: Tensor = self.client_model(img)

            batch_size = intermediate_repr.size(0)
            num_tokens = intermediate_repr.size(1)

            # Use post-processing defense if specified.
            p1 = torch.arange(num_tokens, device=self.rank)
            p2 = torch.argmax(self.position_predictor(intermediate_repr)[:, 1:, 1:], dim=-1)
            # _, p2 = _reorder_tokens(intermediate_repr, None, self.position_predictor, output_permutation=True)
            # _, p2 = _, p2.reshape(batch_size, -1)[:, 1:] - 1

            # predict the patch pixels and measure the performance in [0, 1] domain
            img_pred = self.generate(intermediate_repr)

            # measure the coordinate prediction result with k-class classification
            coordinates_pred = p2

            accuracy.update(p2, torch.arange(num_tokens - 1, device=self.rank).expand(batch_size, -1))

            # decode the classification result to coordinate (i, j)
            row_index = (coordinates_pred // num_patch_per_row).reshape(batch_size, num_patch_per_row, num_patch_per_row) \
                                                               .unsqueeze(-1)
            col_index = (coordinates_pred % num_patch_per_row).reshape(batch_size, num_patch_per_row, num_patch_per_row) \
                                                              .unsqueeze(-1)
            coordinates_pred = torch.cat((row_index, col_index), dim=-1)

            err = (coordinates_pred - coordinates).float().norm(dim=-1, p=1).mean(dim=(1, 2)).sum()
            position_err += err.item()

            # resize the image
            img = F.resize(img, [self._image_size, self._image_size], antialias=True)
            img = self.client_unnormalizer(img).clamp(0, 1)
            img_pred = self.client_unnormalizer(img_pred).clamp(0, 1)

            self.metric.update(img_pred, img)  # pylint: disable=not-callable

        position_err /= len(dataloader.dataset) # type: ignore
        metric = self.metric.compute() | {
            'PositionErr': position_err,
            'PositionPred': accuracy.compute(),
        }

        return { k: float(v) for k, v in metric.items() }

class LitDetokenizer(L.LightningModule):
    _metric: ReconstructionMetric

    def __init__(
        self,
        client_model: nn.Module,
        position_predictor: nn.Module,
        reconstructor: Detokenizer,
        unnormalizer: Callable,
        config: DictConfig,
    ) -> None:
        super().__init__()

        # store hparams to `self._hparams`
        self.save_hyperparameters(ignore=[
            'client_model',
            'position_predictor',
            'reconstructor',
            'unnormalizer',
        ])

        self._config = config
        self._metric = ReconstructionMetric(**dino_image_similarity()).requires_grad_(False)

        self._unnormalizer = unnormalizer
        self._client_model = client_model.eval().requires_grad_(False)

        self.position_predictor = position_predictor
        self.reconstructor = reconstructor

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        return {
            k: v for k, v in super().state_dict().items()
                if 'position_predictor' in k or 'reconstructor' in k
        }

    def configure_optimizers(self):
        training_config = self._config

        optimizer = create_optimizer(
            training_config.optimizer.name,
            [
                { 'params': self.position_predictor.parameters(),
                  'weight_decay': training_config.position_predictor.weight_decay },
                { 'params': self.reconstructor.parameters() },
            ],
            **training_config.optimizer.kwargs,
        )

        scheduler = create_scheduler(
            training_config.scheduler.name,
            optimizer,
            **training_config.scheduler.kwargs
        )

        return OptimizerLRSchedulerConfig(
            optimizer=optimizer,
            lr_scheduler={
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        )

    def forward(self, intermediate_repr: Tensor) -> Tensor:
        return self.reconstructor(intermediate_repr)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, _ = batch

        with torch.no_grad():
            intermediate_repr: Tensor = self._client_model(x)

        # predict the position of the target patch
        position_logits: Tensor = self.position_predictor(intermediate_repr)
        position = torch.arange(0, intermediate_repr.size(1), device=position_logits.device) \
                        .expand(intermediate_repr.size(0), -1)
        acc = (position_logits.argmax(dim=-1) == position).float().mean()

        # predict the patch pixels given the intermediate repr
        intermediate_repr = self.reconstructor.random_mask(
            intermediate_repr, mask_ratio=self._config.mask_ratio
        )

        img_pred: Tensor = self(intermediate_repr)

        reconstruction_loss = nn.functional.mse_loss(img_pred, x)
        prediction_loss = nn.functional.cross_entropy(position_logits, position)

        self.log('Training/Reconstruction Loss', reconstruction_loss, on_step=True, prog_bar=True)
        self.log('Training/Prediction Loss', prediction_loss, on_step=True, prog_bar=True)
        self.log('Training/Accuracy', acc, on_step=True, prog_bar=True)
        return reconstruction_loss + prediction_loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, _ = batch

        intermediate_repr: Tensor = self._client_model(x)
        x_pred: Tensor = self(intermediate_repr)

        self._metric.update(self._unnormalizer(x_pred).clamp(0, 1), self._unnormalizer(x))

    def on_validation_epoch_end(self) -> None:
        self.log_dict({ f'hp/{k}': v.item() for k, v in self._metric.compute().items() })
        self._metric.reset()

        dataset = self.trainer.fit_loop._data_source.instance.dataset # type: ignore

        x = torch.stack([dataset[i][0] for i in range(4)]).to(self.device)
        intermediate_repr: Tensor = self._client_model(x)
        x_pred: Tensor = self(intermediate_repr) # type: ignore

        _, _, w, h = x.size()
        self.logger.experiment.add_image('Reconstruction', # type: ignore
            make_grid(self._unnormalizer(torch.cat([
                x.unsqueeze(1),
                x_pred.unsqueeze(1),
            ], dim=1).view(-1, 3, w, h)), nrow=2),
            self.global_step,
        )

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)

    def on_train_start(self): # pylint: disable=missing-function-docstring
        if self.logger is None:
            return

        score = {
            f'hp/{self._metric.keymap[k]}': torch.tensor(float('nan'))
                for k, _ in self._metric.items(True, False)
        }
        self.logger.log_hyperparams(flatten_dictionary(self._config), score)
