"""
learning_based.py

Data reconstruction task under split inference setting.
"""
import torch
from torch.utils.data import DataLoader, Subset, random_split

from runner.base import TRAIN, VAL
from runner.dataset import create_dataset, create_transforms

from .base import DataReconstructionBase


class LearningBasedDRA(DataReconstructionBase):
    """ Data reconstruction task under split inference setting.

    Threat model:
    - The adversary has black-box access to the target model fc().
    - The adversary has the intermediate repr h = fc(x).
    - The adversary has white-box access to the fs(x).

    Objective:
    - Reconstruct the private image x from the intermediate repr h.
    """
    def _prepare_dataset(self, **kwargs):
        assert not kwargs, f'Unexpected kwargs: {kwargs}.'

        cfg = self.configs
        name = cfg.dataset.name

        preprocess = cfg.preprocess.train
        transform, target_transform = create_transforms(preprocess)
        dataset, collate_fn = create_dataset(
            root=cfg.dataset_dir,
            name=name,
            split='train',
            transform=transform,
            target_transform=target_transform,
            **self.configs.dataset.kwargs
        )

        # We assume that the training split of each dataset is separate as
        # two parts with no overlapping: private parts and public parts.
        private_portion, public_portion = random_split(
            dataset, [0.5, 0.5],
            torch.Generator().manual_seed(0)
        )

        # Attacker should choose `public` portion for the training.
        portion = cfg.dataset.get('portion', None)
        if portion == 'public':
            dataset = public_portion
        elif portion == 'private':
            dataset = private_portion
        else:
            portion = 'all'

        # Drop the remaining samples if the size is specified.
        if (size := cfg.dataset.size) is not None:
            dataset, _ = random_split(
                dataset, [size, len(dataset) - size], # type: ignore
                torch.Generator().manual_seed(0)
            )

        # Use 8:2 for hyperparameter tuning during the training phase.
        self.dataset[TRAIN], self.dataset[VAL] = random_split(
            dataset, [0.8, 0.2], torch.Generator().manual_seed(0))

        self.logger.info(
            'Dataset: %s-%s (train: %d, valid: %d)',
            cfg.dataset.name, portion,
            len(self.dataset[TRAIN]), len(self.dataset[VAL]), # type: ignore
        )

        self.collate_fn = collate_fn
        self.dataloader = {
            split: DataLoader(
                ds, batch_size=cfg.batch_size, shuffle=bool(split == TRAIN),
                num_workers=cfg.workers,
            ) for split, ds in self.dataset.items()
        }

    def _prepare_test_dataset(self, **kwargs):
        assert not kwargs, f'Unexpected kwargs: {kwargs}.'

        cfg = self.configs

        transform, target_transform = create_transforms(cfg.model.preprocess)
        dataset, collate_fn = create_dataset(
            root=cfg.dataset_dir,
            name=cfg.dataset.name,
            split='valid',
            transform=transform,
            target_transform=target_transform,
            **self.configs.dataset.kwargs
        )

        if target := cfg.dataset.get('target', []):
            dataset = Subset(dataset, target)

        self.collate_fn = collate_fn
        self.dataloader = {
            VAL: DataLoader(
                dataset, batch_size=cfg.batch_size, shuffle=False,
                num_workers=cfg.workers,
            )
        }
