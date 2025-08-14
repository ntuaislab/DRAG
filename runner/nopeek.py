"""
nopeek.py

Reference
---------
[1] Vepakomma, P., Singh, A., Gupta, O., & Raskar, R. (2020, November). NoPeek: Information
    leakage reduction to share activations in distributed deep learning. In 2020 International
    Conference on Data Mining Workshops (ICDMW) (pp. 933-942). IEEE.
    https://arxiv.org/abs/2008.09161
"""

# pylint: disable=wrong-import-position
import torch
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.utilities.types import (STEP_OUTPUT,
                                               OptimizerLRSchedulerConfig)
from omegaconf import DictConfig
from torch import Tensor, nn
from torchmetrics.classification import MulticlassAccuracy

from models.clip_classifier import ClientAdapter, ServerAdapter
from models.distance import (DistCorrelation, pairwise_cosine_distances,
                             pairwise_euclidean_distances)
from models.split_network import LOGITS, LitSplitClassifierBase
from runner.utils import create_optimizer, create_scheduler, flatten_dictionary

dmap = {
    'cosine': pairwise_cosine_distances,
    'euclidean': pairwise_euclidean_distances,
}


class NoPeekForClassification(nn.Module):
    def __init__(
        self,
        client: nn.Module,
        server: nn.Module,
    ) -> None:
        super(NoPeekForClassification, self).__init__()

        self._client = client
        self._server = server

    def forward(self, x: Tensor) -> LOGITS:
        return self._server(self._client(x))


class LitNoPeek(LitSplitClassifierBase):
    lamda: float

    def __init__(
        self,
        client: ClientAdapter,
        server: ServerAdapter,
        config: DictConfig,
    ) -> None:
        super().__init__()

        # store hparams to `self._hparams`
        self.save_hyperparameters(ignore=['client', 'server'])

        num_classes = server.head.out_features
        dx = dmap[config.nopeek.dx] # type: ignore
        dz = dmap[config.nopeek.dz] # type: ignore

        self.lamda = config.nopeek.lamda # type: ignore
        self.distance_correlation = DistCorrelation(dx=dx, dz=dz)

        self._config = config
        self._client = client
        self._server = server
        self._metric = MulticlassAccuracy(num_classes=num_classes, average="micro") # type: ignore

    def unwrap(self) -> NoPeekForClassification:
        return NoPeekForClassification(self._client, self._server)

    def configure_optimizers(self):
        training_config = self._config

        optimizer = create_optimizer(
            self._config.optimizer.name,
            [
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
        return self._server(self._client(x))

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch

        h: Tensor = self._client(x)
        distance_correlation_loss = self.distance_correlation(x, h)

        logits: Tensor = self._server(h)
        classification_loss = torch.nn.functional.cross_entropy(logits, y.view(-1))

        loss = classification_loss + self.lamda * distance_correlation_loss

        accuracy = logits.argmax(dim=1).eq(y).float().mean()

        self.log('Train/Distance Correlation Loss', distance_correlation_loss.item(), on_step=True)
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
                                    { 'hp/accuracy': 0.0 })
