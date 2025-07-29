"""
run_likelihood_maximization.py

References
----------
.. [1] Singh, A., Chopra, A., Garza, E., Zhang, E., Vepakomma, P., Sharma, V., & Raskar, R.
       (2021). Disco: Dynamic and invariant sensitive channel obfuscation for deep neural
       networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
       Recognition (pp. 12125-12135).

       https://arxiv.org/pdf/2012.11025
"""

# pylint: disable=wrong-import-position
from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from runner.base import TEST, TRAIN, VAL
from runner.data_reconstruction import OptimizationBasedDRA
from runner.dataset import IterableDataset
from runner.likelihood_maximization import LitLikelihoodMaximization
from runner.utils import save_image


class DummyDataset(IterableDataset):
    def __init__(self, max_steps: int) -> None:
        super().__init__(0, max_steps)


class LikelihoodMaximization(OptimizationBasedDRA):
    """ Inversion task. """
    def run(self, **kwargs) -> None:
        self._prepare_model()
        self._prepare_dataset()

        # Helper variables
        cfg = self.configs

        (intermediate_repr, mask), x_target = self.dataset[TEST]
        assert isinstance(x_target, Tensor)

        pipe = LitLikelihoodMaximization(
            config=self.configs.likelihood_maximization,
            x_target=x_target,
            intermediate_repr=intermediate_repr,
            mask=mask,
            model=self.client_model,
            preprocessor=self.client_preprocessor,
            unnormalizer=self.client_unnormalizer,
        )

        dataloader = {
            TRAIN: DataLoader(DummyDataset(max_steps=cfg.likelihood_maximization.max_steps)),
            VAL: DataLoader(DummyDataset(max_steps=1)),
        }

        logger = TensorBoardLogger(self.working_dir, default_hp_metric=False) if cfg.log_every_n_steps > 0 else False
        trainer = L.Trainer(
            accumulate_grad_batches=1,
            val_check_interval=cfg.val_check_interval if cfg.val_check_interval > 0 else 1,
            check_val_every_n_epoch=None,
            log_every_n_steps=cfg.log_every_n_steps,
            max_steps=cfg.likelihood_maximization.max_steps,
            logger=logger,
            default_root_dir=self.working_dir,
            gradient_clip_algorithm=cfg.likelihood_maximization.gradient_clip_algorithm,
            gradient_clip_val=cfg.likelihood_maximization.gradient_clip_val,
        )
        trainer.fit(
            model=pipe,
            train_dataloaders=dataloader[TRAIN],
            val_dataloaders=dataloader[VAL] if cfg.val_check_interval > 0 else None,
        )

        res = list(self.configs.likelihood_maximization.parameters.image_shape[-2:])
        x: Tensor
        x = pipe.unwrap()
        x = self.client_unnormalizer(v2.functional.resize(x, res, antialias=True)).clamp(0, 1)

        save_image(x, self.working_dir / 'image')

        metric = trainer.validate(
            model=pipe,
            dataloaders=dataloader[VAL],
        )[0]
        print(metric)


@hydra.main(config_path="config",
            config_name=Path(__file__).stem,
            version_base='1.1')
def main(config: DictConfig) -> None:
    # pylint: disable=missing-function-docstring
    task = LikelihoodMaximization(config=config)
    task.run()

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
