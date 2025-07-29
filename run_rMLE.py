"""
run_rMLE.py

References
----------
.. [1] He, Z., Zhang, T., & Lee, R. B. (2019, December). Model inversion attacks
       against collaborative inference. In Proceedings of the 35th Annual Computer
       Security Applications Conference (pp. 148-162).
       https://par.nsf.gov/servlets/purl/10208164
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
from runner.regularize_maximum_likelihood_estimation import \
    LitRegularizeMaximumLikelihoodEstimation
from runner.utils import save_image


class DummyDataset(IterableDataset):
    def __init__(self, max_steps: int) -> None:
        super().__init__(0, max_steps)


class RegularizeMLE(OptimizationBasedDRA):
    """ Inversion task.

    This algorithm directly optimizes the input image to minimize
    the latent distance. The optimization objective function is added
    with regularization terms as prior.
    """
    def run(self, **kwargs) -> None:
        self._prepare_model()
        self._prepare_dataset()

        # Helper variables
        cfg = self.configs

        (intermediate_repr, mask), x_target = self.dataset[TEST]
        assert isinstance(x_target, Tensor)

        pipe = LitRegularizeMaximumLikelihoodEstimation(
            config=self.configs.rMLE,
            x_target=x_target,
            intermediate_repr=intermediate_repr,
            mask=mask,
            model=self.client_model,
            preprocessor=self.client_preprocessor,
            unnormalizer=self.client_unnormalizer,
        )

        dataloader = {
            TRAIN: DataLoader(DummyDataset(max_steps=cfg.rMLE.max_steps)),
            VAL: DataLoader(DummyDataset(max_steps=1)),
        }

        logger = TensorBoardLogger(self.working_dir, default_hp_metric=False) if cfg.log_every_n_steps > 0 else False
        trainer = L.Trainer(
            accumulate_grad_batches=1,
            val_check_interval=cfg.val_check_interval if cfg.val_check_interval > 0 else 1,
            check_val_every_n_epoch=None,
            log_every_n_steps=cfg.log_every_n_steps,
            max_steps=cfg.rMLE.max_steps,
            logger=logger,
            default_root_dir=self.working_dir,
            gradient_clip_algorithm=cfg.rMLE.gradient_clip_algorithm,
            gradient_clip_val=cfg.rMLE.gradient_clip_val,
        )
        trainer.fit(
            model=pipe,
            train_dataloaders=dataloader[TRAIN],
            val_dataloaders=dataloader[VAL] if cfg.val_check_interval > 0 else None,
        )

        res = list(self.configs.rMLE.parameters.image_shape[-2:])
        x: Tensor
        x = pipe.unwrap()
        x = self.client_unnormalizer(v2.functional.resize(x, res, antialias=True)).clamp(0, 1)

        if x.size(0) == 1:
            save_image(x.squeeze(0), self.working_dir / 'image')
        else:
            for tensor, idx in zip(x, self.configs.dataset.target):
                save_image(tensor, self.working_dir / f'{self.configs.dataset.name}_{idx}')

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
    task = RegularizeMLE(config=config)
    task.run()

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
