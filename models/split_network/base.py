from abc import ABC, abstractmethod
from typing import Literal, Tuple, overload

import lightning as L
import torch
from lightning.pytorch.utilities.types import (STEP_OUTPUT,
                                               OptimizerLRSchedulerConfig)
from torch import Tensor, nn
from torchmetrics.classification import Accuracy

# Type alias
LOGITS = Tensor


class Splittable(ABC):
    """ Trait for collaborative inference. """
    @abstractmethod
    def split(
        self,
        split_point: str,
        keep: bool = False,
        output_server_model = False
    ) -> nn.Module | Tuple[nn.Module, nn.Module]:
        """ Split model at the given split point.

        Arguments
        ---------
        split_point : str
            Split point.

        keep : bool
            Keep the remaining layer.

        Returns
        -------
        model
            Model instance.
        """


class SplittableIdentity(torch.nn.Identity, Splittable):
    """ Identity layer for collaborative inference. """
    @overload
    def split(
        self,
        split_point: str,
        keep: bool,
        output_server_model: Literal[False]
    ) -> nn.Module:
        ...

    @overload
    def split(
        self,
        split_point: str,
        keep: bool,
        output_server_model: Literal[True]
    ) -> Tuple[nn.Module, nn.Module]:
        ...

    def split(
        self,
        split_point: str,
        keep: bool = False,
        output_server_model = False
    ) -> nn.Module | Tuple[nn.Module, nn.Module]:
        try:
            if getattr(self, '_splitted') is True:
                raise RuntimeError("Model is already splitted.")
        except AttributeError:
            pass

        self._splitted = True
        return (self, nn.Identity()) if output_server_model else self


class LitSplitClassifierBase(L.LightningModule):
    """ Adapter between Transformers CLIP and Torch Lightning. """
    _metric: Accuracy

    @abstractmethod
    def forward(self, x: Tensor) -> LOGITS:
        ...

    @abstractmethod
    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        ...

    @abstractmethod
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch

        loss = torch.nn.functional.cross_entropy(self(x), y.view(-1))
        self.log('Train/Loss', loss.item(), on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch

        preds = self(x).argmax(dim=-1)
        self._metric.update(preds, y)

    def on_validation_epoch_end(self):
        self.log('hp/accuracy', self._metric.compute(), on_epoch=True)
        self._metric.reset()

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch

        preds = self(x).argmax(dim=-1)
        self._metric.update(preds, y)

    def on_test_epoch_end(self):
        self.log('hp/accuracy', self._metric.compute(), on_epoch=True)
        self._metric.reset()
