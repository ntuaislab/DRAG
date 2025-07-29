"""
pretrained.py

Function for loading a fine-tuned model from local checkpoint.
"""

import json
from pathlib import Path
from typing import Tuple

import lightning as L
from clip.model import ModifiedResNet
from omegaconf import DictConfig, OmegaConf
from torch import nn
from transformers import CLIPVisionModelWithProjection, Dinov2Model

from models import create_model, split_model
from models.clip_classifier import ClientAdapter, ServerAdapter
from runner.disco import LitDisco
from runner.nopeek import LitNoPeek
from runner.utils import parse_torch_dtype


def from_pretrained(
    path : Path | str,
    unwrap_model : bool = True
) -> L.LightningModule | Tuple[nn.Module, nn.Module]:
    """ Load a fine-tuned model from the local checkpoint.

    Parameters
    ----------
    path : Path | str
        The path to the local checkpoint.

    split_model : bool
        Whether to split the model into client and server parts. Default is True.
        Note that some components would be removed if the model is split.

    Returns
    -------
    Tuple[nn.Module, nn.Module]
        The client and server models.
    """
    if isinstance(path, str):
        path = Path(path)

    with open(path / 'model_index.json') as f:
        _class_name = json.load(f)['_class_name']

    with open(path / '.hydra' / 'config.yaml') as f:
        cfg = OmegaConf.load(f)

        assert isinstance(cfg, DictConfig)
        assert cfg.dataset.name == 'imagenet', f'Unknown dataset: {cfg.dataset.name}'
        dtype = parse_torch_dtype(cfg.model.get('torch_dtype', 'float32'))
        out_features = 1000

    # Load base model and probe with classification head
    model = create_model(cfg.model.name, cfg.model.checkpoint, torch_dtype=dtype)
    client, server = split_model(model, cfg.model.name, cfg.model.split_points, True) # type: ignore
    assert client is not None and server is not None

    if isinstance(server, CLIPVisionModelWithProjection):
        server.visual_projection = nn.Linear(server.visual_projection.in_features, out_features)
    elif isinstance(model, ModifiedResNet):
        server = nn.Sequential(server, nn.Linear(model.output_dim, out_features))
    elif isinstance(model, Dinov2Model):
        server = nn.Sequential(server, nn.Linear(model.config.hidden_size, out_features))
    else:
        raise ValueError(f'Unsupported model: {cfg.model.name}')

    # Load the model from the checkpoint, then update the target model parameters,
    # or even replace the model with a new one (because it might has different
    # architecture and forwarding method)
    match _class_name:
        case 'NoPeek':
            model = LitNoPeek.load_from_checkpoint(
                path / 'last.ckpt',
                config=cfg,
                client=ClientAdapter(client),
                server=ServerAdapter(server), # type: ignore
            )

        case 'DISCO':
            if isinstance(model, CLIPVisionModelWithProjection | Dinov2Model):
                model = LitDisco.load_from_checkpoint(
                    path / 'last.ckpt',
                    config=cfg,
                    client=ClientAdapter(client),
                    server=ServerAdapter(server), # type: ignore
                    strict=False, # skip loading the reconstructor
                )

            else:
                raise ValueError(f'Unknown model type {type(model)}')

        case _:
            raise ValueError(f'Unknown model class {_class_name}')

    if unwrap_model:
        return model._client.client, model._server.server

    return model
