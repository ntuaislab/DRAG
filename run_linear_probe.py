"""
run_linear_probe.py
-------------------

Linear probing foundation models on ImageNet-1K dataset.
"""

# pylint: disable=wrong-import-position
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from models import create_model
from models.split_network import LOGITS, LitSplitClassifierBase
from runner.base import TRAIN, VAL
from runner.classifier import ClassifierRunner
from runner.utils import create_optimizer, parse_torch_dtype


class LitClassificationHead(LitSplitClassifierBase):
    def __init__(
        self,
        fc: nn.Linear,
    ) -> None:
        super().__init__()

        self.fc = fc
        self._metric = MulticlassAccuracy(
            num_classes=fc.out_features, average="micro",
        ) # type: ignore

    def unwrap(self) -> nn.Linear:
        return self.fc

    def forward(self, x: Tensor) -> LOGITS:
        return self.fc(x)


class LinearProbeTrainer(ClassifierRunner):
    """ Trainer for the classification task, w/ privacy defense. """
    def _prepare_dataset(self, **kwargs) -> None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load cache (see also: cache.py)
        cfg = self.configs

        root = Path.cwd() / 'checkpoints' / 'embedding' / cfg.model.checkpoint / cfg.dataset.name

        self.dataset = {
            TRAIN: (
                torch.load(root / "train" / "embeds_part01.pt", weights_only=True).to(device),
                torch.load(root / "train" / "labels_part01.pt", weights_only=True).to(device)
            ),
            VAL: (
                torch.load(root / "val" / "embeds.pt", weights_only=True).to(device),
                torch.load(root / "val" / "labels.pt", weights_only=True).to(device)
            ),
        }

    def _prepare_model(self, **kwargs) -> None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Helper variables
        cfg = self.configs

        # Initialize the vision model
        model = create_model(cfg.model.name, cfg.model.checkpoint,
                             parse_torch_dtype(cfg.model.torch_dtype))

        self.model = model.to(device)

    def run(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._prepare_dataset()
        self._prepare_model()

        # Helper variables
        cfg = self.configs

        # Solve the linear regression problem w/ PyTorch
        x, y = self.dataset[TRAIN]
        feature_dim = x.size(1)

        perf = {}
        decay = self.configs.weight_decay
        for lamba in reversed(np.logspace(decay.start, decay.stop, decay.num, base=10)):
            fc = nn.Linear(feature_dim, 1000).to(device)
            optimizer = create_optimizer(
                cfg.optimizer.name,
                [
                    { 'params': fc.parameters() },
                ],
                weight_decay=lamba,
                **cfg.optimizer.kwargs,
            )

            self.logger.info(f'Start training {lamba=}... ')
            pbar = tqdm(range(cfg.max_steps), ncols=0, desc=f"lambda={lamba:.3g}")
            x, y = self.dataset[TRAIN]
            for _ in pbar:
                optimizer.zero_grad()
                y_hat: Tensor = fc(x)
                loss = nn.functional.cross_entropy(y_hat, y)
                acc = y_hat.argmax(dim=1).eq(y).float().mean()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=loss.item(), acc=acc.item())

            with torch.no_grad():
                x, y = self.dataset[VAL]
                acc = fc(x).argmax(dim=1).eq(y).float().mean()

            fc = fc.cpu()
            torch.save(fc.state_dict(), f"head_{lamba=:.3g}.pt")

            perf[lamba] = acc.item()
            self.logger.info(f"Training done! {lamba=:.3g} ({acc:.2%})")

        best_lamba = max(perf, key=perf.get) # type: ignore
        self.logger.info(f"Best lambda: {best_lamba} ({perf[best_lamba]:.2%})")
        self.logger.info(perf)


@hydra.main(config_path="config",
            config_name=Path(__file__).stem,
            version_base='1.1')
def main(config: DictConfig) -> None: # pylint: disable=missing-function-docstring
    LinearProbeTrainer(config=config).run()

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
