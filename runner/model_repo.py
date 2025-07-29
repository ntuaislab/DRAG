"""
model_repo.py
"""

from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from models.detokenizer import Detokenizer

def detokenizer(ckpt_dir: str | Path) -> Detokenizer:
    """ Load the detokenizer model from the checkpoint directory.

    Arguments
    ---------
    ckpt_dir : str or Path
        Path to the checkpoint directory.

    Returns
    -------
    Detokenizer
        The detokenizer model.
    """
    if isinstance(ckpt_dir, str):
        ckpt_dir = Path(ckpt_dir)

    with open(Path(ckpt_dir) / 'model_index.json', 'r', encoding='utf-8') as f: # pylint: disable=invalid-name
        model_config = OmegaConf.load(f)
        assert isinstance(model_config, DictConfig)

    assert model_config.pop('_class_name') == 'PatchReconstructor'
    model = Detokenizer(**model_config) # type: ignore

    state_dict = torch.load(
        ckpt_dir / 'last.ckpt',
        map_location=next(model.parameters()).device,
        weights_only=False,
    )['state_dict']
    model.load_state_dict(
        state_dict={
            k.replace('reconstructor.', ''): v
                for k, v in state_dict.items()
                    if k.startswith('reconstructor.')
        },
        strict=False
    )

    return model
