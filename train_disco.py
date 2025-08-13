"""
train_disco.py

References
----------
[1] Singh, A., Chopra, A., Garza, E., Zhang, E., Vepakomma, P., Sharma, V., & Raskar, R. (2021).
    Disco: Dynamic and invariant sensitive channel obfuscation for deep neural networks.
    In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
    (pp. 12125-12135).

[2] aidecentralized/InferenceBenchmark
    https://github.com/aidecentralized/InferenceBenchmark
"""

# pylint: disable=wrong-import-position
from functools import singledispatch
from pathlib import Path
from typing import Dict

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, random_split

from models import create_model
from models.clip_classifier import ClientAdapter, ServerAdapter
from models.detokenizer import Detokenizer
from models.split_network import Splittable
from models.split_network.split_clip import SplittableCLIP
from models.split_network.split_dinov2 import SplittableDINOv2
from runner.base import TRAIN, VAL
from runner.classifier import ClassifierRunner
from runner.disco import LitDisco
from runner.model_repo import detokenizer
from runner.utils import parse_torch_dtype


def prediction_head(ckpt_dir: str | Path) -> nn.Linear:
    if isinstance(ckpt_dir, str):
        ckpt_dir = Path(ckpt_dir)

    with open(ckpt_dir / 'model_index.json', 'r') as f:
        model_config = OmegaConf.load(f)
        assert isinstance(model_config, DictConfig)

    assert model_config.pop('_class_name') == 'Linear'

    head = nn.Linear(**model_config) # type: ignore
    head.load_state_dict(
        torch.load(ckpt_dir / "head.pt", map_location=next(head.parameters()).device)
    )
    return head


@singledispatch
def set_pipe(model: Splittable, head: nn.Linear, cfg: DictConfig) -> L.LightningModule:
    raise NotImplementedError(f"Unsupported model: {type(model)}")


@set_pipe.register(SplittableCLIP)
def _set_pipe_clip_vit(model: SplittableCLIP, head: nn.Linear, cfg: DictConfig) -> L.LightningModule:
    server: SplittableCLIP

    client, server = model.split(cfg.model.split_points, output_server_model=True) # type: ignore
    assert server is not None
    server.visual_projection = head

    # TODO: Load a channel pruner
    # self.pipe._channel_pruner.load_state_dict(torch.load(assets / "channel_pruner.pt"))

    return LitDisco(
        client=ClientAdapter(client),
        server=ServerAdapter(server),
        reconstructor=detokenizer(cfg.disco.pretrained_reconstructor),
        config=cfg, # type: ignore
    )


@set_pipe.register(SplittableDINOv2)
def _set_pipe_dinov2(model: SplittableDINOv2, head: nn.Linear, cfg: DictConfig) -> L.LightningModule:
    client, server = model.split(cfg.model.split_points, output_server_model=True) # type: ignore
    assert server is not None
    server = nn.Sequential(server, head)

    # TODO: Load a channel pruner
    # self.pipe._channel_pruner.load_state_dict(torch.load(assets / "channel_pruner.pt"))

    return LitDisco(
        client=ClientAdapter(client),
        server=ServerAdapter(server),
        reconstructor=detokenizer(cfg.disco.pretrained_reconstructor),
        config=cfg, # type: ignore
    )


class DiscoTrainer(ClassifierRunner):
    """ Trainer for the classification task, w/ privacy defense. """
    pipe: L.LightningModule
    reconstructor: Detokenizer

    def _prepare_dataset(self, **kwargs) -> None:
        super()._prepare_dataset(**kwargs)

        # We assume that the training split of each dataset is separate as
        # two parts with no overlapping: private parts and public parts.
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
        self.logger.info(f"Loaded the head from {(head_ckpt := cfg.model.pretrained_head)}")
        self.logger.info(f"Loaded the reconstructor from {cfg.disco.pretrained_reconstructor}")

        self.pipe = set_pipe(model, prediction_head(head_ckpt), cfg)

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
            # monitor="Validation/Accuracy",
            every_n_train_steps=cfg.checkpoint.every_n_train_steps,
        )

        trainer = L.Trainer(
            accumulate_grad_batches=1,
            check_val_every_n_epoch=None,
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

    task = DiscoTrainer(config=config)
    task.run()

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
