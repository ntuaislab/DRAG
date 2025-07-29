"""
disco.py
--------

Impl of privacy defense DISCO.
"""

# pylint: disable=wrong-import-position
import math
from copy import deepcopy
from pathlib import Path
from typing import Dict

import torch
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.utilities.types import (STEP_OUTPUT,
                                               OptimizerLRSchedulerConfig)
from omegaconf import DictConfig
from torch import Tensor, nn
from torchmetrics.classification import MulticlassAccuracy
from torchvision.transforms.v2.functional import resize
from transformers import CLIPVisionModelWithProjection, Dinov2Model

from models import create_model, split_model
from models.clip_classifier import ClientAdapter, ServerAdapter
from models.detokenizer import Detokenizer
from models.split_network import LOGITS, LitSplitClassifierBase, Splittable
from models.functional import invert_gradient

from .utils import create_optimizer, create_scheduler, flatten_dictionary


class Preprocessor(nn.Module):
    use_spatial_decoupling: bool

    def __init__(self, conv: nn.Conv2d, use_spatial_decoupling: bool = False) -> None:
        """
        Preprocessor network for spatial decoupling.

        Arguments
        ---------
        conv: nn.Conv2d
            Convolutional layer to be used as the preprocessor.

        spatial_decoupling: bool
            Whether to apply spatial decoupling or not. If True, the input tensor
        """
        super().__init__()

        self.use_spatial_decoupling = use_spatial_decoupling
        self.conv = conv
        self.p = int(math.sqrt(conv.out_channels))

        assert (self.p * self.p == conv.out_channels), \
            f"Invalid number of channels: {conv.out_channels}"

    def forward(
        self,
        x: Tensor,
        decoupling : bool | None = None
    ) -> Tensor:
        if not (decoupling := self.use_spatial_decoupling if decoupling is None else decoupling):
            return self.conv(x)

        bs, c, h, w = x.size()
        patch_size = h // self.p
        assert (h % self.p == 0), f"Invalid patch size: {self.p}"

        x = x.unfold(2, patch_size, patch_size) \
             .unfold(3, patch_size, patch_size) \
             .reshape(-1, 3, patch_size, patch_size)

        x = resize(x, [224, 224])
        x = self.conv(x)

        _, c, h, w = x.size()
        x = x.reshape(bs, self.p * self.p, c, h, w) \
             .mean(dim=1)

        return x


def _normalize_score(score: Tensor, r: float, temperature: float) -> Tensor:
    num_prunable_channels = int(score.size(-1) * r)

    sorted_score, _ = torch.sort(score, dim=-1)
    mean = sorted_score[:, num_prunable_channels].unsqueeze(-1)

    return ((score - mean) / temperature).sigmoid()


class DiscoForClassification(nn.Module, Splittable):
    def __init__(
        self,
        client: nn.Module,
        server: nn.Module,
        channel_pruner: nn.Module,
        pruning_ratio: float = 0.5,
        temperature: float = 30.0,
    ) -> None:
        super(DiscoForClassification, self).__init__()

        self._client = client
        self._server = server
        self._channel_pruner = channel_pruner
        self.pruning_ratio = pruning_ratio
        self.temperature = temperature

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    # TODO: Fix the return value
    @property
    def config(self) -> Dict:
        return self._client.client.config

    def split(
        self,
        split_point: str,
        keep: bool = False,
        output_server_model = False
    ) -> 'DiscoForClassification':
        assert split_point == 'client', f"Invalid split point: {split_point}"

        if output_server_model:
            core = self._server.server
            self._server = nn.Identity()
            return self, core

        self._server = nn.Identity()
        return self

    def forward(self, pixel_values: Tensor) -> LOGITS:
        h: Tensor = self._client(pixel_values)

        s: Tensor = self._channel_pruner(h)
        s = _normalize_score(s, self.pruning_ratio, self.temperature)
        s = s[:, None, :]

        logits: Tensor = self._server(s * h)
        return logits


class LitDisco(LitSplitClassifierBase):
    rho: float
    temperature: float
    pruning_ratio: float

    def __init__(
        self,
        client: ClientAdapter,
        server: ServerAdapter,
        config: DictConfig,
        reconstructor: Detokenizer | None = None,
    ) -> None:
        super().__init__()

        # store hparams to `self._hparams`
        self.save_hyperparameters(ignore=['client', 'server', 'reconstructor'])

        num_classes = server.head.out_features

        # Infer tensor shape w/ dummy input
        fake_data = torch.randn(1, 3, 224, 224)

        intermediate_repr: Tensor = client(fake_data)
        _, _, d = intermediate_repr.size()

        # Preprocessor (Dynamic switchable):
        # client[0] = Preprocessor(client[0], use_spatial_decoupling=config.disco.use_spatial_decoupling)

        # Channel pruner:
        # Similar to the server model, but with different number of prediction heads
        channel_pruner = deepcopy(server)
        if isinstance(channel_pruner.server, CLIPVisionModelWithProjection):
            channel_pruner.server.visual_projection = nn.Linear(d, d)
        else:
            channel_pruner.server[-1] = nn.Linear(d, d)

        # Decoder:
        # Reconstruct the input image from the intermediate representation
        self.rho = config.disco.rho
        self.temperature = config.disco.temperature
        self.pruning_ratio = config.disco.pruning_ratio

        self._config = config
        self._client = client
        self._server = server
        self._metric = MulticlassAccuracy(num_classes=num_classes) # type: ignore
        self._reconstructor = reconstructor
        self._channel_pruner = channel_pruner

    def unwrap(self) -> DiscoForClassification:
        return DiscoForClassification(
            self._client,
            self._server,
            self._channel_pruner,
            self.pruning_ratio,
            self.temperature,
        )

    def configure_optimizers(self):
        training_config = self._config

        optimizer = create_optimizer(
            self._config.optimizer.name,
            [
                { 'params': self._channel_pruner.parameters() },
                { 'params': self._reconstructor.parameters() },
                { 'params': self._client.parameters() },
                { 'params': self._server.parameters() },
            ],
            **training_config.optimizer.kwargs,
        )

        scheduler = create_scheduler(
            training_config.scheduler.name, optimizer, **training_config.scheduler.kwargs
        )

        return OptimizerLRSchedulerConfig(
            optimizer=optimizer,
            lr_scheduler={
                "scheduler": scheduler,
            },
        )

    def forward(self, x: Tensor) -> LOGITS:
        h: Tensor = self._client(x)

        s: Tensor = self._channel_pruner(h)
        s = _normalize_score(s, self.pruning_ratio, self.temperature)
        s = s[:, None, :]
        h = s * h

        logits: Tensor = self._server(h)
        return logits

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch

        h: Tensor = self._client(x)

        s: Tensor = self._channel_pruner(h.detach())
        s = _normalize_score(s, self.pruning_ratio, self.temperature)
        s = s[:, None, :]

        reconstruction: Tensor = self._reconstructor(invert_gradient(s) * h.detach())
        logits: Tensor = self._server(s * h)

        classification_loss = torch.nn.functional.cross_entropy(logits, y.view(-1))
        reconstruction_loss = torch.nn.functional.mse_loss(reconstruction, x)
        loss = reconstruction_loss + self.rho * classification_loss

        accuracy = logits.argmax(dim=1).eq(y).float().mean()

        self.log('Train/Reconstruction Loss', reconstruction_loss.item(), on_step=True)
        self.log('Train/Classification Loss', classification_loss.item(), on_step=True)
        self.log('Train/Accuracy', accuracy.item(), on_step=True, prog_bar=True)
        self.log('Train/Loss', loss.item(), on_step=True, logger=False, prog_bar=True)

        return loss

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)

    def on_train_start(self):
        if self.logger is None:
            return

        self.logger.log_hyperparams(flatten_dictionary(self._config),
                                    { 'hp/accuracy': torch.tensor(float('nan')) })


def load_pruner(
    config: DictConfig,
    ckpt_dir: Path,
) -> nn.Module:
    """
    Load the NoPeek model from the checkpoint

    Arguments
    ---------
    config : DictConfig
        Configuration for the model

    ckpt_dir : Path | None
        Directory to load the checkpoint from

    Returns
    -------
    L.LightningModule
        DISCO model
    """
    model = create_model(config.model.name, config.model.checkpoint)
    client, server = split_model(model, config.model.name, config.model.split_points, True) # type: ignore

    assert isinstance(client, CLIPVisionModelWithProjection | Dinov2Model)
    assert isinstance(server, CLIPVisionModelWithProjection | Dinov2Model)

    fake_data = torch.randn(1, 3, 224, 224)

    intermediate_repr: Tensor = client(fake_data)
    _, _, d = intermediate_repr.size()

    if isinstance(server, Dinov2Model):
        server = nn.Sequential(server, nn.Linear(d, d))

    server = ServerAdapter(server)
    channel_pruner = deepcopy(server)

    if isinstance(channel_pruner.server, CLIPVisionModelWithProjection):
        channel_pruner.server.visual_projection = nn.Linear(d, d)
    else:
        channel_pruner.server[-1] = nn.Linear(d, d)

    state_dict = torch.load(ckpt_dir, map_location=client.device, weights_only=False)['state_dict']
    channel_pruner.load_state_dict({
        k.replace('_channel_pruner.', ''): v
            for k, v in state_dict.items()
                if k.startswith('_channel_pruner.')
    })

    return channel_pruner
