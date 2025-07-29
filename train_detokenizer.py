"""
train_detokenizer.py

Outline
-------
Script to train an inverter f^{-1} that decodes the intermediate repr h to x.

Reference
----------
[1] Darcet, T., Oquab, M., Mairal, J., & Bojanowski, P. (2023). Vision
    transformers need registers. arXiv preprint arXiv:2309.16588.

[2] He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022).
    Masked autoencoders are scalable vision learners. In Proceedings of the
    IEEE/CVF conference on computer vision and pattern recognition (pp. 16000-16009).
"""

# pylint: disable=wrong-import-position
import json
from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader

from runner.base import TRAIN, VAL
from runner.detokenization import DetokenizationRunner, LitDetokenizer
from runner.utils import default_num_workers


class DetokenizationTrainer(DetokenizationRunner):
    @property
    def _create_checkpoint_folder(self) -> bool:
        return True

    @property
    def _create_writer(self) -> bool:
        return False

    def run(self) -> None:
        self._prepare_dataset()
        self._prepare_model()

        # Helper variables
        cfg = self.configs

        # Create dataloader for further training.
        dataloader = {
            TRAIN: DataLoader(
                self.dataset[TRAIN], batch_size=cfg.batch_size,
                shuffle=True, num_workers=cfg.workers, collate_fn=self.collate_fn,
            ),
            VAL: DataLoader(
                self.dataset[VAL], batch_size=cfg.batch_size,
                shuffle=False, num_workers=cfg.workers, collate_fn=self.collate_fn,
            )
        }
        lr_monitor = LearningRateMonitor(logging_interval='step')
        default_root_dir = self.checkpoint_dir if self._create_checkpoint_folder else self.working_dir
        checkpoint_callback = ModelCheckpoint(
            dirpath=default_root_dir,
            save_top_k=cfg.checkpoint.save_top_k,
            save_last=True,
            every_n_train_steps=cfg.checkpoint.every_n_train_steps,
        )

        trainer = L.Trainer(
            # Max training steps
            max_steps=cfg.max_steps,
            # Training tricks
            accumulate_grad_batches=cfg.accumulate_grad_batches,
            gradient_clip_algorithm=cfg.gradient_clip_algorithm,
            gradient_clip_val=cfg.gradient_clip_val,
            # Validation
            check_val_every_n_epoch=None,
            val_check_interval=cfg.val_check_interval,
            limit_val_batches=cfg.get('limit_val_batches', None),
            # Logging
            log_every_n_steps=cfg.log_every_n_steps,
            callbacks=[
                checkpoint_callback, # type: ignore
                lr_monitor,
            ],
            logger=TensorBoardLogger(
                self.working_dir,
                default_hp_metric=False,
            ), # type: ignore
            fast_dev_run=False,
            benchmark=False,
            default_root_dir=default_root_dir,
        )
        model = LitDetokenizer(
            client_model=self.client_model,
            position_predictor=self.position_predictor,
            reconstructor=self.reconstructor,
            unnormalizer=self.client_unnormalizer,
            config=cfg, # type: ignore
        )
        with open(self.working_dir / 'model_index.json', 'w') as f:
            json.dump(self.reconstructor.config, f, indent=4)

        trainer.fit(
            model=model,
            train_dataloaders=dataloader[TRAIN],
            val_dataloaders=dataloader[VAL],
        )


@hydra.main(config_path="config",
            config_name=Path(__file__).stem,
            version_base='1.1')
def main(config: DictConfig) -> None: # pylint: disable=missing-function-docstring
    if config.get('workers', -1) == -1:
        n_workers = default_num_workers()
        with open_dict(config):
            config.workers = n_workers

    task = DetokenizationTrainer(config=config)
    task.run()

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
