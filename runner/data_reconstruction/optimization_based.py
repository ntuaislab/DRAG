"""
optimization_based.py
"""

import torch
from omegaconf import ListConfig

from runner.base import TEST
from runner.dataset import create_dataset, create_transforms

from .base import DataReconstructionBase


class OptimizationBasedDRA(DataReconstructionBase):
    def _prepare_dataset(self, **kwargs):
        assert not kwargs, f'Unexpected kwargs: {kwargs}.'
        assert getattr(self, 'client_model', None) is not None, \
            'Client model must be prepared before dataset preparation.'

        # We need the client model to extract the intermediate representation.
        # Therefore, we swap the order of the model preparation and dataset preparation.
        config = self.configs
        target = self.configs.dataset.target

        transform, target_transform = create_transforms(config.model.preprocess)
        dataset, _ = create_dataset(
            config.dataset_dir, config.dataset.name, split='valid',
            transform=transform, target_transform=target_transform,
            **config.dataset.kwargs)

        # Select the target image(s)
        if isinstance(target, ListConfig):
            x_target = [dataset[t][0].unsqueeze(0) for t in target] # Triggered dataset.transform()
            x_target = torch.cat(x_target, dim=0)
        elif isinstance(target, int):
            x_target, _ = dataset[target]                           # Triggered dataset.transform()
            x_target = x_target.unsqueeze(0)
        else:
            raise ValueError(f'Invalid target: {target}. Must be int or ListConfig.')

        # Load required images and remove unused
        x_target = x_target.to(self.device)
        intermediate_repr, mask = self._forward(x_target)

        x_target = self.client_unnormalizer(x_target).clamp(0, 1)   # Reverse the normalization
        self.dataset[TEST] = (intermediate_repr, mask), x_target
