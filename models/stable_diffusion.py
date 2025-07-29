"""
stable_diffusion.py

Utility functions for models.
"""

import torch
from torch import Tensor
from transformers import CLIPTextModel, CLIPTokenizer


@torch.no_grad()
def get_unconditioned_embedding(
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    normalize: bool = False
) -> Tensor:
    """ Forward to get the unconditional embedding from the text encoder.

    Arguments
    ---------
    text_encoder : CLIPTextModel
        CLIPTextModel instance.

    tokenizer : CLIPTokenizer
        CLIPTokenizer instance.

    normalize : bool
        Whether to normalize the output by LayerNorm.

    Returns
    -------
    Tensor
        Unconditioned embedding.
    """
    input_ids = tokenizer(
        "",
        padding='max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).input_ids.to(text_encoder.device)

    assert "CLIPTextModel" == text_encoder.config.architectures[0]

    # Take the un-normalized (layernorm and l2-normalized) token embeddings
    out = text_encoder.text_model(
        input_ids=input_ids,
        output_hidden_states=True
    )

    if normalize:
        return out.last_hidden_state

    return out.hidden_states[-1]
