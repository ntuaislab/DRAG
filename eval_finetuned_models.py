"""
eval_finetuned_models.py
"""

import json
from pathlib import Path
from typing import Callable

import hydra
import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Metric
from torchmetrics.classification import MulticlassAccuracy
from transformers import CLIPVisionModelWithProjection

from models import create_model, split_model
from runner.base import VAL
from runner.classifier import ClassifierRunner
from runner.data_reconstruction import apply_defense
from runner.dataset import IMAGENET1K_CATEGORIES
from runner.pretrained import from_pretrained
from runner.utils import default_num_workers, parse_torch_dtype


class LitAdapter(L.LightningModule):
    """ Adapter between Transformers CLIP and Torch Lightning. """
    def __init__(
        self,
        client: CLIPVisionModelWithProjection,
        server: CLIPVisionModelWithProjection,
        defense: Callable[[Tensor], Tensor],
    ) -> None:
        super().__init__()

        self._client = client.eval()
        self._server = server.eval()
        self._defense = defense
        self._metric: Metric = MulticlassAccuracy(
            num_classes=self._server.visual_projection.out_features,
            average="micro",
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._server(self._defense(self._client(x))).image_embeds

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch

        intermediate_repr: Tensor = self._client(pixel_values=x)
        intermediate_repr: Tensor = self._defense(intermediate_repr)
        logits: Tensor = self._server(intermediate_repr).image_embeds
        preds = logits.argmax(dim=-1)
        self._metric.update(preds, y)

    def on_test_epoch_end(self):
        self.log('Test/Accuracy', self._metric.compute(), prog_bar=True)
        self._metric.reset()


class ClassificationRunner(ClassifierRunner):
    """ Trainer for the classification task. """
    def run(self):
        model_config = self.configs.model
        defense_config = self.configs.defense

        # Wrap the model into a Lightning module
        assert isinstance((dataset := self.dataset[VAL]), Dataset)
        dataloader = DataLoader(
            dataset, batch_size=self.configs.batch_size, shuffle=False,
            num_workers=self.configs.workers, drop_last=False,
        )
        trainer = L.Trainer(
            deterministic=True,
            benchmark=False,
            logger=False,
        )

        checkpoint = model_config.checkpoint
        if checkpoint is not None and (Path(checkpoint)).exists():
            model = from_pretrained(checkpoint, unwrap_model=False) # type: ignore
            self.logger.info('Loaded fine-tuned model from %s', checkpoint)
        else:
            dtype = parse_torch_dtype(model_config.torch_dtype)
            split_points= model_config.split_points

            model = create_model(model_config.name, checkpoint, torch_dtype=dtype)
            client, server = split_model(model, model_config.name, split_points, output_server_model=True)
            assert isinstance(server, CLIPVisionModelWithProjection)

            assets = (self.running_dir / "assets" / "linear_probe" / model_config.checkpoint / 'imagenet')
            server.visual_projection = nn.Linear(server.visual_projection.in_features, len(IMAGENET1K_CATEGORIES))
            server.visual_projection.load_state_dict(
                torch.load(assets / "head.pt", map_location=server.visual_projection.weight.device)
            )
            self.logger.info('Loaded linear probe from %s', assets / "head.pt")

            model = LitAdapter(
                client, # type: ignore
                server, # type: ignore
                lambda x: apply_defense(x, defense_config.name, **defense_config.kwargs),
            )

        assert isinstance(model, L.LightningModule)
        metric = trainer.test(model=model, dataloaders=dataloader, verbose=True)[0]
        self.logger.info(metric)

        with open(self.working_dir / 'metrics.json', 'w') as f:
            json.dump({
                'model': OmegaConf.to_object(model_config),
                'defense': OmegaConf.to_object(defense_config),
                'metric': metric,
            }, f, indent=4)


@hydra.main(config_path="config",
            config_name=Path(__file__).stem,
            version_base='1.1')
def main(config: DictConfig) -> None: # pylint: disable=missing-function-docstring
    if config.get('workers', -1) == -1:
        n_workers = default_num_workers()
        with open_dict(config):
            config.workers = n_workers

    task = ClassificationRunner(config=config)
    task.logger.info("Using %d workers.", task.configs.workers)
    task.run()

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
