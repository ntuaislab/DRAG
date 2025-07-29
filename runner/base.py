"""
base.py

Base class for all tasks.
"""

import logging
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict

import hydra
import torch
from omegaconf import DictConfig, open_dict
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .utils import flatten_dictionary, same_seeds

Metric = Dict
TRAIN, VAL, TEST = 'train', 'val', 'test'


class RunnerBase(ABC):
    """ Base class for all runners. """
    # Tracking execution status
    logger: logging.Logger

    # Environment and run-time variables
    start_time: float
    checkpoint_dir: Path
    running_dir: Path
    working_dir: Path

    configs: DictConfig

    # Initialize member field for dataset.
    dataset: Dict[str, Dataset | Tensor]
    dataloader: Dict[str, DataLoader]
    collate_fn: Callable | None

    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        """
        Arguments
        ---------
        dry_run : bool
            If True, does not save checkpoints and tensorboard logs.

        use_cpu : bool
            If True, enable trainer using cpu when cuda is unavailable.
        """
        # Initialize logger.
        self.logger = self._initialize_logger()

        try:
            running_dir = Path(hydra.utils.get_original_cwd())
        except ValueError: # In case that hydra is not used.
            running_dir = Path.cwd()
        self.running_dir = running_dir
        self.working_dir = Path.cwd()
        self.logger.info("Running dir: %s", self.running_dir)
        self.logger.info("Working dir: %s", self.working_dir)

        self.configs = config

        # Record start time.
        self.start_time = time.time()

        # Create directory for saving checkpoints.
        self._initialize_recorder()

        # Configure random seed to ensure reproducibility.
        seed = self.configs.get('seed', 0)
        with open_dict(self.configs):
            self.configs.seed = seed

        self.logger.info('Setting seed to %d.', seed)
        same_seeds(seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize member field for dataset.
        self.dataset = {}
        self.dataloader = {}
        self.collate_fn = None

    @property
    def _create_checkpoint_folder(self) -> bool:
        """ Property to determine whether to create checkpoint folder.

        Derived class can override this property to change the behavior.
        """
        return False

    @property
    def name(self) -> str:
        return re.sub(r'(?<!^)(?=[A-Z])', '_', self.__class__.__name__).lower()

    @property
    def hparams(self) -> Dict:
        return flatten_dictionary(self.configs)

    def _initialize_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        return logger

    def _makedirs(self, *args: Path, exist_ok=True):
        for path in args:
            path.mkdir(parents=True, exist_ok=exist_ok)

    def _initialize_recorder(self):
        # Prepare checkpoint directory.
        if not self._create_checkpoint_folder:
            return

        timestamp: str = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(self.start_time))

        ckpt_root: Path = Path(self.configs.checkpoint_dir)
        dataset: str = self.configs.dataset.name

        self.checkpoint_dir = ckpt_root / self.name / dataset / timestamp

        self._makedirs(self.checkpoint_dir, exist_ok=True)

        # Create a symbolic link from the working directory to the checkpoint directory.
        (self.working_dir / 'checkpoints').symlink_to(self.checkpoint_dir,
                                                        target_is_directory=True)


    @abstractmethod
    def _prepare_dataset(self, **kwargs):
        ...

    @abstractmethod
    def run(self):
        ...
