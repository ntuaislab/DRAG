"""
train_nopeek.py

References
----------
[1] Vepakomma, P., Singh, A., Gupta, O., & Raskar, R. (2020, November). NoPeek:
    Information leakage reduction to share activations in distributed deep learning.
    In 2020 International Conference on Data Mining Workshops (ICDMW) (pp. 933-942). IEEE.

[2] aidecentralized/InferenceBenchmark
    https://github.com/aidecentralized/InferenceBenchmark
"""

# pylint: disable=wrong-import-position
from pathlib import Path
from typing import Dict

import hydra
import lightning as L
import torch
from clip.model import ModifiedResNet
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader, random_split
from transformers import CLIPVisionModelWithProjection, Dinov2Model

from models import create_model, split_model
from models.clip_classifier import ClientAdapter, ServerAdapter
from runner.base import TRAIN, VAL
from runner.classifier import ClassifierRunner
from runner.dataset import IMAGENET1K_CATEGORIES
from runner.nopeek import LitNoPeek
from runner.utils import parse_torch_dtype


class NoPeekTrainer(ClassifierRunner):
    """ Trainer for the classification task, w/ privacy defense. """
    pipe: L.LightningModule

    def _prepare_dataset(self, **kwargs) -> None:
        super()._prepare_dataset(**kwargs)

        # We assume that the training split of each dataset is separate as
        # two parts with no overlapping: private parts and public parts.
        if True: # TODO: Support the public-private split
            dataset, _ = random_split(
                self.dataset[TRAIN], [0.5, 0.5], torch.Generator().manual_seed(0))

        # Use 8:2 for hyperparameter tuning during the training phase.
        self.dataset[TRAIN], self.dataset[VAL] = random_split(
            dataset, [0.8, 0.2], torch.Generator().manual_seed(0))

    def _prepare_model(self, **kwargs) -> None:
        # Helper variables
        cfg = self.configs

        # Initialize the target model
        model = create_model(cfg.model.name, cfg.model.checkpoint,
                             parse_torch_dtype(cfg.model.torch_dtype))

        # Load trained head from the linear probe process
        head_ckpt = (
            Path.home()
            / "assets"
            / "linear_probe"
            / cfg.model.checkpoint
            / cfg.dataset.name
        )
        state_dict = torch.load(
            head_ckpt / "head.pt",
            map_location=next(model.parameters()).device,
            weights_only=True,
        )
        self.logger.info(f"Loaded the linear probe head from {head_ckpt}")

        # For now, we only support specific models from transformers and
        # the ModifiedResNet from clip.
        if isinstance(model, CLIPVisionModelWithProjection):
            head = nn.Linear(model.visual_projection.in_features, len(IMAGENET1K_CATEGORIES))
            head.load_state_dict(state_dict)
            model.visual_projection = head

            client, server = model.split(cfg.model.split_points, output_server_model=True)

        elif isinstance(model, ModifiedResNet):
            head = nn.Linear(model.output_dim, len(IMAGENET1K_CATEGORIES))
            head.load_state_dict(state_dict)

            client, server = split_model(
                model, cfg.model.name, cfg.model.split_points, output_server_model=True
            ) # type: ignore
            server = nn.Sequential(server, head)

        elif isinstance(model, Dinov2Model):
            head = nn.Linear(model.config.hidden_size , len(IMAGENET1K_CATEGORIES))
            head.load_state_dict(state_dict)

            client, server = split_model(
                model, cfg.model.name, cfg.model.split_points, output_server_model=True
            )
            server = nn.Sequential(server, head)

        else:
            raise ValueError(f"Unsupported model: {cfg.model.name}")

        self.pipe = LitNoPeek(
            client=ClientAdapter(client),
            server=ServerAdapter(server),
            config=cfg, # type: ignore
        )

    def run(self):
        self._prepare_dataset()
        self._prepare_model()

        dataloader = {
            TRAIN: DataLoader(
                self.dataset[TRAIN], batch_size=self.configs.batch_size, shuffle=True,
                num_workers=self.configs.workers, drop_last=True,
            ),
            VAL: DataLoader(
                self.dataset[VAL], batch_size=self.configs.batch_size, shuffle=False,
                num_workers=self.configs.workers, drop_last=False,
            ),
        }

        cfg = self.configs
        default_root_dir = self.checkpoint_dir if self._create_checkpoint_folder else self.working_dir

        checkpoint_callback = ModelCheckpoint(
            dirpath=default_root_dir / "checkpoint",
            save_last=True,
            save_top_k=cfg.checkpoint.save_top_k,
            every_n_train_steps=cfg.checkpoint.every_n_train_steps,
        )

        trainer = L.Trainer(
            accumulate_grad_batches=cfg.accumulate_grad_batches,
            val_check_interval=cfg.val_check_interval,
            log_every_n_steps=cfg.log_every_n_steps,
            max_steps=cfg.max_steps,
            callbacks=[
                checkpoint_callback,
            ],
            logger=TensorBoardLogger(
                self.working_dir,
                default_hp_metric=False,
            ),
            fast_dev_run=False,
            deterministic='warn',
            benchmark=False,
            default_root_dir=default_root_dir,
            gradient_clip_algorithm='norm',
            gradient_clip_val=10.0,
        )

        if ckpt_path := cfg.get('ckpt_path', None):
            self.logger.info(f"Resuming from checkpoint: {cfg.ckpt_path}")

        trainer.fit(
            model=self.pipe,
            train_dataloaders=dataloader[TRAIN],
            val_dataloaders=dataloader[VAL],
            ckpt_path=ckpt_path,
        )
        metric = trainer.test(
            model=self.pipe,
            dataloaders=dataloader[VAL],
        )[0]
        print(metric)


@hydra.main(config_path="config",
            config_name=Path(__file__).stem,
            version_base='1.1')
def main(config: DictConfig) -> None: # pylint: disable=missing-function-docstring
    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)

    task = NoPeekTrainer(config=config)
    task.run()

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
