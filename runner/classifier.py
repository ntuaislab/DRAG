"""
classifier.py
-------------

Classifier runner.
"""

import torchmetrics
from omegaconf import DictConfig

from runner.dataset import create_dataset, create_transforms

from .base import TRAIN, VAL, RunnerBase


class ClassifierRunner(RunnerBase):
    """ Classification task. """

    metric: torchmetrics.Metric

    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        super().__init__(config=config)
        self._prepare_dataset()

    def _prepare_dataset(self, **kwargs):
        assert not kwargs, f'Unexpected kwargs: {kwargs}.'

        cfg = self.configs
        name = cfg.dataset.name

        preprocess = cfg.preprocess.train
        transform, target_transform = create_transforms(preprocess)
        self.dataset[TRAIN], collate_fn = create_dataset(
            root=self.configs.dataset_dir,
            name=name,
            split='train',
            transform=transform,
            target_transform=target_transform,
            **self.configs.dataset.kwargs
        )

        preprocess = cfg.preprocess.validation
        transform, target_transform = create_transforms(preprocess)
        self.dataset[VAL], _ = create_dataset(
            root=self.configs.dataset_dir, name=name, split='valid',
            transform=transform, target_transform=target_transform,
            **self.configs.dataset.kwargs
        )

        self.collate_fn = collate_fn
