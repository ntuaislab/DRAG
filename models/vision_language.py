"""
vision_language.py
"""

from typing import List

import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPModel, CLIPProcessor


@torch.no_grad()
def get_classification_head(
    ckpt: str,
    cls_name: List[str],
    merge_projection_head: bool = False,
) -> nn.Linear:
    """ Get the k-class classification head from the text model.

    Arguments
    ---------
    ckpt : str
        Checkpoint path of the CLIP model.

    cls_name : List[str]
        List of class names.

    merge_projection_head : bool
        Whether to merge the projection head.

        If True, the text embedding matrix will be combined with the vision
        projection head via commutative property: W(Px) = (WP)x. Then the
        vision projection head can be discarded.
    """
    processor = CLIPProcessor.from_pretrained(ckpt)
    assert isinstance(processor, CLIPProcessor), "Processor should be of type CLIPProcessor"
    assert getattr(processor, "tokenizer", None) is not None, "Processor should have an image processor"

    model = CLIPModel.from_pretrained(ckpt)
    input_ids = processor.tokenizer( # type: ignore
        [f'a photo of a {name}' for name in cls_name],
        padding="max_length",
        max_length=processor.tokenizer.model_max_length, # type: ignore
        truncation=True,
        return_tensors="pt",
    ).input_ids
    text_embeds = model.get_text_features(input_ids=input_ids)
    text_embeds = F.normalize(text_embeds, p=2, dim=-1)

    if merge_projection_head:
        head = nn.Linear(model.vision_model.config.hidden_size, len(cls_name), bias=True)
        head.weight.data = text_embeds @ model.visual_projection.weight.data
    else:
        head = nn.Linear(model.config.projection_dim, len(cls_name), bias=True)
        head.weight.data = text_embeds

    head.bias.data.zero_()

    return head

