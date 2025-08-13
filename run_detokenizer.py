"""
run_detokenizer.py
"""

# pylint: disable=wrong-import-position
from pathlib import Path
from pprint import pprint
from tqdm import tqdm

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from torchvision.utils import save_image

from models.distance import dino_image_similarity
from runner.base import VAL
from runner.data_reconstruction import ReconstructionMetric
from runner.dataset import CLIP_IMG_UNNORMALIZE
from runner.detokenization import DetokenizationRunner


@torch.no_grad()
@hydra.main(config_path="config",
            config_name="default",
            version_base='1.1')
def main(config: DictConfig) -> None:
    origin: str = Path(config.ckpt_dir) / '.hydra' / 'config.yaml'
    origin: DictConfig = OmegaConf.load(origin)

    # Patch the config
    with open_dict(origin):
        origin.world_size = 1
        origin.dataset_dir = config.dataset_dir
        origin.checkpoint_dir = config.checkpoint_dir
        origin.batch_size = 64

        if (dataset_cfg := config.get('dataset')) is not None:
            origin.dataset.name = dataset_cfg.name
            origin.dataset.kwargs = dataset_cfg.kwargs

        if (defense_cfg := config.get('defense')) is not None:
            origin.defense = defense_cfg

        if (adaptive_attack_cfg := config.get('adaptive_attack')) is not None:
            origin.adaptive_attack = adaptive_attack_cfg

        if (batch_size := config.get('batch_size')) is not None:
            origin.batch_size = batch_size

    # Setup the runner
    task = DetokenizationRunner(config=origin)
    task._prepare_test_dataset()
    task._prepare_model()
    task.load_checkpoint(config.ckpt_dir)

    # Standard evaluation
    task.metric = ReconstructionMetric(**dino_image_similarity()).to(task.device)
    metric = task.evaluate()
    metric = pd.Series(metric)
    metric.to_json(Path(config.ckpt_dir) / 'metric.json', indent=4)

    print(f"{task.configs.model.split_points} - ({config.ckpt_dir})")
    print(metric)

    # Demonstration
    if (targets := config.dataset.get('target', None)) is None:
        return

    root = Path('outputs') / 'inverse_network'
    root.mkdir(exist_ok=True, parents=True)
    dataset = task.dataloader[VAL].dataset
    for target, (img, *_) in tqdm(zip(targets, dataset)):
        img = img.to(task.device).unsqueeze(0)

        intermediate_repr = task.client_model(img)
        img_pred = task.generate(intermediate_repr)
        img_pred = CLIP_IMG_UNNORMALIZE(img_pred).clamp(0, 1)

        # Create the output directory
        outdir = root / origin.model.checkpoint / f'{origin.dataset.name}.{target}' / origin.model.split_points
        outdir.mkdir(exist_ok=True, parents=True)
        (outdir / '.hydra').mkdir(exist_ok=True)

        # Create .hydra/config.yaml and provide some information
        sample_config = OmegaConf.create(origin)
        with open_dict(sample_config):
            sample_config.dataset.target = target

        with open(outdir / '.hydra' / 'config.yaml', 'w') as f:
            OmegaConf.save(sample_config, f)

        save_image(img_pred, outdir / 'image.png', normalize=False)


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
