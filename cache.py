"""
cache.py
--------
Script to cache embeddings from CLIP/DINOv2 models.

Arguments
---------
ckpt: Checkpoint name of the CLIP model.
"""

from copy import deepcopy
from pathlib import Path

import torch
from omegaconf import OmegaConf, open_dict
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers.models.clip.modeling_clip import \
    CLIPVisionModelWithProjection

from runner.dataset import create_dataset, create_transforms


@torch.no_grad()
def cache(ckpt: str):
    """
    Cache image embeddings from CLIP/DINOv2 models.

    root/embedding
    └── <checkpoint>
        └── <dataset>
            ├── train
            │   ├── embeds_part01.pt  # private set, client uses it to fine-tune the model
            │   ├── embeds_part02.pt  # private set, client uses it to validate the model
            │   ├── embeds_part03.pt  # public set, adversary uses it to invert the data
            │   ├── embeds_part04.pt  # public set, adversary uses it to validate the **inversion model** (if needed)
            │   ├── labels_part01.pt
            │   ├── labels_part02.pt
            │   ├── labels_part03.pt
            │   └── labels_part04.pt
            └── val
                ├── embeds_part01.pt  # private set, client uses it to fine-tune the model
                ├── embeds_part02.pt  # private set, client uses it to validate the model
                ├── embeds_part03.pt  # public set, adversary uses it to invert the data
                ├── embeds_part04.pt  # public set, adversary uses it to validate the **inversion model** (if needed)
                ├── labels_part01.pt
                ├── labels_part02.pt
                ├── labels_part03.pt
                └── labels_part04.pt
    """

    config_str = f"""
checkpoint_dir: ./checkpoints/  # NOTE: Update this path if needed
dataset_dir: ./datasets/        # NOTE: Update this path if needed
workers: 16
batch_size: 128

model:
  name: CLIPVisionModelWithProjection
  torch_dtype: float32
  checkpoint: {ckpt}
  image_size: 224
  preprocess:
  - clip_vit_processor: {{}}

dataset:
  name: imagenet
"""
    config = OmegaConf.create(config_str)

    device = torch.device('cuda')

    ckpt_dir: Path = (
        Path(config.checkpoint_dir)
        / 'embedding'
        / config.model.checkpoint # First level
        / config.dataset.name     # Second level
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / 'train').mkdir(parents=True, exist_ok=True)
    (ckpt_dir / 'val').mkdir(parents=True, exist_ok=True)

    dataset_config = deepcopy(config.dataset)
    with open_dict(dataset_config):
        dataset_config.dataset_dir = config.dataset_dir

    print(f'Checkpoint: {ckpt}')
    print(f'Target dir: {ckpt_dir}')
    # Use `create_model` for DINOv2
    # model = create_model(config.model.name, config.model.checkpoint,
    #                      parse_torch_dtype(config.model.torch_dtype)).to(device)
    model = CLIPVisionModelWithProjection.from_pretrained(config.model.checkpoint).to(device)

    # Disable next line for DINOv2
    model.visual_projection = nn.Identity() # type: ignore
    hidden_dim = model.config.hidden_size

    transform, _ = create_transforms(OmegaConf.to_object(config.model.preprocess)) # type: ignore
    train_dataset, _ = create_dataset(
        dataset_config.dataset_dir, dataset_config.name, split='train',
        transform=transform,
    )

    val_dataset, _ = create_dataset(
        dataset_config.dataset_dir, dataset_config.name, split='val',
        transform=transform,
    )

    private_dataset, public_dataset = random_split(
        train_dataset, [0.5, 0.5], torch.Generator().manual_seed(0))

    # Partition 01 and 02 for private set, 03 and 04 for public set
    private_train_dataset, private_val_dataset = random_split(
        private_dataset, [0.8, 0.2], torch.Generator().manual_seed(0))

    public_train_dataset, public_val_dataset = random_split(
        public_dataset, [0.8, 0.2], torch.Generator().manual_seed(0))

    for idx, dataset in enumerate([private_train_dataset,
                                   private_val_dataset,
                                   public_train_dataset,
                                   public_val_dataset], 1):
        if ((ckpt_dir / 'train' / f'embeds_part{idx:02d}.pt').exists()
            and (ckpt_dir / 'train' / f'labels_part{idx:02d}.pt').exists()
        ):
            print(f'Dataset {dataset_config.name} (train) part {idx:02d} already exists. Skip.')
            continue

        dataloader = DataLoader(dataset, batch_size=config.batch_size, num_workers=config.workers)
        f = torch.empty(len(dataset), hidden_dim, dtype=torch.float32, device=device)
        Y = torch.empty(len(dataset), dtype=torch.long, device=device)

        start = 0
        for im, label in tqdm(dataloader, desc=f'{dataset_config.name}{idx:02d}', ncols=0):
            im = im.to(device)
            bs = im.size(0)
            f[start:start+bs] = model(im).image_embeds
            Y[start:start+bs] = label
            start += bs

        f, Y = f.cpu(), Y.cpu()
        torch.save(f, ckpt_dir / 'train' / f'embeds_part{idx:02d}.pt')
        torch.save(Y, ckpt_dir / 'train' / f'labels_part{idx:02d}.pt')

    if (ckpt_dir / 'val' / 'embeds.pt').exists() and (ckpt_dir / 'val' / 'labels.pt').exists():
        print(f'Dataset {dataset_config.name} (validation) part already exists. Skip.')
        return

    dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.workers)
    f = torch.empty(len(val_dataset), hidden_dim, dtype=torch.float32, device=device) # type: ignore
    Y = torch.empty(len(val_dataset), dtype=torch.long, device=device) # type: ignore

    start = 0
    for im, label in tqdm(dataloader, desc=f'{dataset_config.name}{idx:02d}', ncols=0):
        im = im.to(device)
        bs = im.size(0)
        f[start:start+bs] = model(im)
        Y[start:start+bs] = label
        start += bs

    f, Y = f.cpu(), Y.cpu()
    torch.save(f, ckpt_dir / 'val' / 'embeds.pt')
    torch.save(Y, ckpt_dir / 'val' / 'labels.pt')


if __name__ == "__main__":
    cache('openai/clip-vit-base-patch16')
