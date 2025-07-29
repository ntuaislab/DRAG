"""
roll_dice.py

Script to determine evaluation target, served for reproducibility purposes.
"""
import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf, open_dict
from tabulate import tabulate

from runner.dataset import create_dataset, create_transforms
from runner.utils import flatten_dictionary, same_seeds

CONFIG = OmegaConf.create(
"""
dataset_dir: ./datasets
workers: 0

# List datasets to be evaluated.
dataset:
  mscoco:
    year: 2017
  ffhq: {}
  imagenet: {}

# For visualization purposes.
transform:
  - resize:
      size: 224
  - center_crop:
      size: 224
"""
)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0,
                    help="Seed value for reproducibility.")
parser.add_argument('--num', type=int, default=15,
                    help="Number of samples to select.")
parser.add_argument('--split', type=str, default='valid', choices=['train', 'valid'],
                    help="Dataset split to evaluate.")
parser.add_argument('--outdir', type=str, default=None,
                    help="Export the selected samples to target directory.")
args = parser.parse_args()

with open_dict(CONFIG):
    CONFIG.update(vars(args))

print(tabulate(flatten_dictionary(CONFIG).items(),
               headers=['Key', 'Value'],
               tablefmt='fancy_grid'))

def main():
    same_seeds(args.seed)

    outdir = Path(args.outdir) if args.outdir is not None else None

    transform, _ = create_transforms(CONFIG.transform)
    for dataset_name, dataset_kwargs in CONFIG.dataset.items():
        dataset, _ = create_dataset(
            root=CONFIG.dataset_dir,
            name=dataset_name,
            split=args.split,
            transform=transform,
            **dataset_kwargs
        )
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))
        indices = sorted(indices[:args.num].numpy().tolist())

        print()
        print(f"Dataset: {dataset_name} (split: {args.split}, size: {len(dataset)})\n"
              "Indices:\n"
              f"\t{indices}")

        if outdir is None:
            continue

        for idx in indices:
            img, _ = dataset[idx]
            img.save(outdir / f"{dataset_name}_{idx:06d}.jpg")

if __name__ == '__main__':
    main()
