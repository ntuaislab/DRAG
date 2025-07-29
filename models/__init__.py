"""
models/__init__.py
==================

Functions to create and split models.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple, overload

import clip
import torch
from clip.model import ModifiedResNet
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import \
    StableDiffusionImg2ImgPipeline
from omegaconf import DictConfig
from torch import Tensor, nn
from transformers import CLIPVisionModelWithProjection, Dinov2Model

from .distance import AdaptiveCosineSimilarityLoss, CosineSimilarityLoss
from .pipeline_diffusion_img2img import DDPMImg2ImgPipeline
from .split_network import SplittableCLIP, SplittableDINOv2


def use_criterion(
    name: str,
    **kwargs,
) -> nn.Module:
    """ Return criterion instance given class name.

    Arguments
    ---------
    name : str
        Name of criterion.

    Returns
    -------
    criterion : nn.Module
        Criterion instance.

    Raises
    ------
    ValueError
        If dataset name is not recognized.
    """
    match name:
        case 'CosineSimilarityLoss':
            criterion = CosineSimilarityLoss(**kwargs)

        case 'AdaptiveCosineSimilarityLoss':
            criterion = AdaptiveCosineSimilarityLoss(**kwargs)

        case _ if (ctor := getattr(nn, name, None)) is not None:
            criterion = ctor()

        case _:
            raise ValueError(f'Unknown criterion: {name}')

    return criterion


def create_model(
    name: str,
    weights: str,
    torch_dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """ Return model instance given class name.

    Arguments
    ---------
    name : str
        Name of model.

    weights : str
        Path to weights.

    Returns
    -------
    model
        Model instance.

    Raises
    ------
    ValueError
        If dataset name is not recognized.
    """
    match name:
        case 'Dinov2Model':
            model = SplittableDINOv2.from_pretrained(weights, torch_dtype=torch_dtype)

        case 'CLIPVisionModelWithProjection':
            model = SplittableCLIP.from_pretrained(weights, torch_dtype=torch_dtype)

        case 'ModifiedResNet':
            # NOTE: auto converted to float32 while loading weights to CPU
            model, _ = clip.load(weights.split('/')[-1], device='cpu')
            model = model.visual

        case _:
            raise ValueError(f'Unknown model: {name}')

    return model.eval() # type: ignore


class ToDtype(nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.dtype)


@overload
def _split_modified_resnet(
    model: ModifiedResNet,
    split_point: str | int,
    output_server_model: Literal[False]
) -> Tuple[nn.Module, None]:
    ...


@overload
def _split_modified_resnet(
    model: ModifiedResNet,
    split_point: str | int,
    output_server_model: Literal[True]
) -> Tuple[nn.Module, nn.Module]:
    ...


def _split_modified_resnet(
    model: ModifiedResNet,
    split_point: str | int,
    output_server_model: bool = False
) -> Tuple[nn.Module, nn.Module | None]:
    if isinstance(split_point, str):
        split_point = {
            'layer_0': 10,
            'layer_1': 11,
            'layer_2': 12,
            'layer_3': 13,
            'layer_4': 14,
            'image_embeds': 15
        }[split_point]

    to_dtype = ToDtype(next(model.parameters()).dtype)
    layers = list(model.children())
    client = nn.Sequential(to_dtype, *layers[:split_point])
    server = nn.Sequential(*layers[split_point:]) if output_server_model else None

    return client, server


@overload
def split_model(
    model: nn.Module,
    name: str,
    split_point: str,
    output_server_model: Literal[True]
) -> Tuple[nn.Module, nn.Module]:
    ...


@overload
def split_model(
    model: nn.Module,
    name: str,
    split_point: str,
    output_server_model: Literal[False]
) -> Tuple[nn.Module, None]:
    ...


def split_model(
    model: nn.Module,
    name: str,
    split_point: str,
    output_server_model: bool = False,
) -> Tuple[nn.Module, nn.Module | None]:
    """ Split model at the given split point.

    Arguments
    ---------
    model: nn.Module
        Model to be splitted.

    name: str
        Name of the model.

    split_point: str
        Split point.

    output_server_model: bool
        If True, return the server model also.

    Returns
    -------
    model
        Splitted model.

    Raises
    ------
    ValueError
        If model name is not recognized.
    """
    client: nn.Module
    server: nn.Module | None = None

    match name:
        case 'ModifiedResNet':
            assert isinstance(model, ModifiedResNet)
            client, server = _split_modified_resnet(
                model, split_point, output_server_model=output_server_model
            )

        case 'Dinov2Model':
            assert isinstance(model, SplittableDINOv2)
            if output_server_model:
                client, server = model.split(split_point, keep=False, output_server_model=True)
            else:
                client = model.split(split_point, keep=False, output_server_model=False)

        case 'CLIPVisionModelWithProjection':
            assert isinstance(model, SplittableCLIP)
            if output_server_model:
                client, server = model.split(split_point, keep=False, output_server_model=True)
            else:
                client = model.split(split_point, keep=False, output_server_model=False)

        case _:
            raise ValueError(f'Unknown model: {name}')

    if output_server_model:
        return client, server

    return client, None


def make_split(
    model: nn.Module,
    head_ckpt: Path | os.PathLike,
    model_name: str,
    split_point: str,
):
    n_classes = 1000

    # Load trained head from the linear probe process
    state_dict = torch.load(
        head_ckpt,
        map_location=next(model.parameters()).device,
        weights_only=True,
    )

    # For now, we only support specific models from transformers and
    # the ModifiedResNet from clip.
    if isinstance(model, CLIPVisionModelWithProjection):
        head = nn.Linear(model.visual_projection.in_features, n_classes)
        head.load_state_dict(state_dict)
        model.visual_projection = head

        client, server = model.split(split_point, output_server_model=True)

    elif isinstance(model, ModifiedResNet):
        head = nn.Linear(model.output_dim, n_classes)
        head.load_state_dict(state_dict)

        client, server = split_model(
            model, model_name, split_point, output_server_model=True
        ) # type: ignore
        server = nn.Sequential(server, head)

    elif isinstance(model, Dinov2Model):
        head = nn.Linear(model.config.hidden_size, n_classes)
        head.load_state_dict(state_dict)

        client, server = split_model(
            model, model_name, split_point, output_server_model=True
        )
        server = nn.Sequential(server, head)

    else:
        raise ValueError(f"Unsupported model: {type(model)}")

    return client, server


@dataclass
class AutoencoderKLConfig:
    scaling_factor: float


class FakeAutoencoderKL(nn.Module):
    @property
    def config(self) -> AutoencoderKLConfig:
        return AutoencoderKLConfig(scaling_factor=1.0)

    # def encode(self, x: Tensor, return_dict: bool = True) -> AutoencoderKLOutput:
    #     if not return_dict:
    #         return (x, )
    #
    #     return AutoencoderKLOutput(x)

    def decode(self, x: Tensor, return_dict: bool = True, generator = None) -> DecoderOutput:
        if not return_dict:
            return (x, )

        return DecoderOutput(sample=x)


def create_image_prior(
    cfg: DictConfig,
    dtype: torch.dtype = torch.float32,
    shrink_variance: bool = False,
):
    ckpt = cfg.checkpoint
    pipe: DiffusionPipeline
    match cfg.name:
        case 'StableDiffusionImg2ImgPipeline':
            assert not shrink_variance

            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                ckpt, torch_dtype=dtype, safety_checker=None,
            )

        case 'DiffusionModel' | 'DM':
            pipe = DiffusionPipeline.from_pretrained(
                ckpt, torch_dtype=dtype, safety_checker=None,
            )
            unet = pipe.unet
            scheduler = pipe.scheduler

            pipe = DDPMImg2ImgPipeline(unet=unet, scheduler=scheduler)
            pipe.vae = FakeAutoencoderKL()

        case _:
            raise ValueError(f"Invalid image prior: {cfg.name}")

    pipe.set_progress_bar_config(disable=True)
    return pipe
