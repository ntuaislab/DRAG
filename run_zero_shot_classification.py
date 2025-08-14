"""
run_zero_shot_classification.py

Eval zero-shot classification on CLIP models, support defenses.
"""

import os;

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
from pathlib import Path
from typing import Callable

import hydra
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Metric
from torchmetrics.classification import MulticlassAccuracy

from models.split_network import SplittableCLIP
from models.vision_language import get_classification_head
from runner.base import VAL
from runner.classifier import ClassifierRunner
from runner.data_reconstruction import apply_defense
from runner.dataset import IMAGENET1K_CATEGORIES
from runner.utils import default_num_workers


class LitAdapter(L.LightningModule):
    """ Adapter between Transformers CLIP and Torch Lightning. """
    def __init__(
        self,
        client: nn.Module,
        server: nn.Module,
        head: nn.Linear,
        defense: Callable[[Tensor], Tensor],
    ) -> None:
        super().__init__()

        self._head = head.eval()
        self._client = client.eval()
        self._server = server.eval()
        self._defense = defense
        self._metric: Metric = MulticlassAccuracy(
            num_classes=self._head.out_features,
            average='micro',
        )

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch

        intermediate_repr: Tensor = self._client(pixel_values=x)
        intermediate_repr: Tensor = self._defense(intermediate_repr)
        image_embeds: Tensor = self._server(intermediate_repr).image_embeds
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        logits: Tensor = self._head(image_embeds)
        preds = logits.argmax(dim=-1)
        self._metric.update(preds, y)

    def on_test_epoch_end(self):
        self.log('Test/Accuracy', self._metric.compute(), prog_bar=True)
        self._metric.reset()


class ZeroShotClassificationRunner(ClassifierRunner):
    """ Trainer for the classification task. """
    def run(self):
        cfg = self.configs

        self._prepare_dataset()
        assert isinstance((dataset := self.dataset[VAL]), Dataset), \
            f'Expected dataset[VAL] to be a Dataset, got {type(dataset)}.'

        # Load text model to construct the classification head
        head = get_classification_head(
            cfg.model.checkpoint, IMAGENET1K_CATEGORIES, merge_projection_head=True,)

        # Wrap the model into a Lightning module
        dataloader = DataLoader(
            dataset, batch_size=self.configs.batch_size, shuffle=False,
            num_workers=self.configs.workers, drop_last=False,
        )
        trainer = L.Trainer(
            deterministic=True,
            benchmark=False,
            logger=False,
        )

        clip: SplittableCLIP = SplittableCLIP.from_pretrained(cfg.model.checkpoint)
        client, server = clip.split(cfg.model.split_points, keep=False, output_server_model=True)
        server.visual_projection = nn.Identity()
        model = LitAdapter(
            client, server, head,
            lambda x: apply_defense(x, cfg.defense.name, **cfg.defense.kwargs)
        )
        metric = trainer.test(model=model, dataloaders=dataloader, verbose=True)[0]
        self.logger.info(metric)

        with open(self.working_dir / 'metrics.json', 'w') as f:
            json.dump({
                'model': OmegaConf.to_object(cfg.model),
                'defense': OmegaConf.to_object(cfg.defense),
                'metric': metric,
            }, f)


@hydra.main(config_path="config",
            config_name=Path(__file__).stem,
            version_base='1.1')
def main(config: DictConfig) -> None: # pylint: disable=missing-function-docstring
    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    if config.get('workers', -1) == -1:
        n_workers = default_num_workers()
        with open_dict(config):
            config.workers = n_workers

    ZeroShotClassificationRunner(config=config).run()


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
