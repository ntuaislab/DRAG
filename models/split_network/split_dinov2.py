"""
split_dinov2.py

Base class for implementing model splitting.
"""

from copy import deepcopy
from types import MethodType
from typing import Literal, Optional, Tuple, overload

import torch
from torch import Tensor, nn
from transformers import Dinov2Model

from .base import Splittable


class NoOp(nn.Module):
    """ Identity module for returning the input tensor as is.
    Issues: https://github.com/pytorch/pytorch/issues/42015
    """
    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        return x


def _client_tokenization_forward(
    self: Dinov2Model,
    pixel_values: torch.Tensor
) -> Tensor:
    return self.embeddings(pixel_values)


def _client_intermediate_repr_forward(
    self: Dinov2Model,
    pixel_values: torch.Tensor,
    bool_masked_pos: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Tensor:
    embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

    encoder_outputs = self.encoder(
        embedding_output,
        head_mask=head_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    sequence_output = encoder_outputs[0]
    return sequence_output


def _client_embedding_forward(
    self: Dinov2Model,
    pixel_values: torch.Tensor
) -> Tensor:
    # NOTE:
    # l2-normalization is occurred when doing cosine loss similarity with text embeds.
    # See transformers.CLIPModel

    embedding_output = self.embeddings(pixel_values)                # [bs, N, d_hidden]
    encoder_outputs = self.encoder(embedding_output)
    sequence_output = encoder_outputs[0]
    sequence_output = self.layernorm(sequence_output)
    pooled_output = sequence_output[:, 0, :]
    pooled_output = pooled_output.unsqueeze(1)                      # [bs, 1, d_out]
    return pooled_output


def _client_split_dinov2(
    model: Dinov2Model,
    split_point: str
) -> Dinov2Model:
    if split_point == 'embeddings':
        # 'embeddings':
        # convert the pixels as tokens, then normalized with LayerNorm

        # Stub out all layers except embedding
        # model.encoder.pre_layrnorm = nn.Identity()
        model.encoder = nn.Identity()
        model.layernorm = nn.Identity()
        model.forward = MethodType(_client_tokenization_forward, model)

        return model

    if split_point.startswith('encoder_layer'):
        # 'encoder_layer_<n>':
        # convert the pixels as tokens, and pass through the first n layers of the encoder
        split_idx = int(split_point.split('_')[-1])
        if split_idx > len(model.encoder.layer):
            raise ValueError(f"Invalid split point: {split_point}")

        model.encoder.layer = nn.ModuleList(
            model.encoder.layer[:split_idx]
        )
        # Stub out all remaining layers
        model.layernorm = nn.Identity()
        model.forward = MethodType(_client_intermediate_repr_forward, model)

        return model

    if split_point == 'image_embeds':
        # 'image_embeds':
        # pass through the entire model except the normalization
        model.forward = MethodType(_client_embedding_forward, model)

        return model

    raise ValueError(f"Invalid split point: {split_point}")


def _server_intermediate_repr_forward(
    self: Dinov2Model,
    intermediate_repr: Tensor,
    bool_masked_pos: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Tensor:
    # embedding_output = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

    encoder_outputs = self.encoder(                                         # hidden_states:
        intermediate_repr,                                                  #   [bs, N, d_hidden]
        head_mask=head_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    sequence_output = encoder_outputs[0]
    sequence_output = self.layernorm(sequence_output)
    pooled_output = sequence_output[:, 0, :]                                # pooled_output:
                                                                            #   [bs, d_hidden]
    return pooled_output


def _server_split_dinov2(
    model: Dinov2Model,
    split_point: str
) -> Dinov2Model:
    if split_point == 'embeddings':
        raise NotImplementedError

    if split_point.startswith('encoder_layer'):
        # 'encoder_layer_<n>':
        # convert the pixels as tokens, and pass through the first n layers of the encoder

        split_idx = int(split_point.split('_')[-1])
        if split_idx > len(model.encoder.layer):
            raise ValueError(f"Invalid split point: {split_point}")

        model.embeddings = NoOp()
        model.encoder.layer = nn.ModuleList(
            model.encoder.layer[split_idx:]
        )

        model.forward = MethodType(_server_intermediate_repr_forward, model)
        return model

    # if split_point == 'image_embeds':
    #     return nn.Identity()

    raise ValueError(f"Invalid split point: {split_point}")


class SplittableDINOv2(Dinov2Model, Splittable):
    """ Run Dinov2Model with collaborative inference mode. """
    def _split_keep(self, split_point: str) -> None:
        # pylint: disable=invalid-name
        match split_point:
            case 'embeddings':
                split_idx = 0
            case 'image_embeds':
                split_idx = -1
            case _ if split_point.startswith('encoder_layer'):
                split_idx = int(split_point.split('_')[-1])
                if split_idx > len(self.encoder.layer):
                    raise ValueError(f"Invalid split point: {split_point}")
            case _:
                raise ValueError(f"Invalid split point: {split_point}")

        # overwrite definition of the forward method
        def forward(
            self: SplittableDINOv2,
            hidden_states: Tensor
        ) -> Tuple[Tensor, Tuple[Tensor]]:
            vision_outputs = self.encoder(hidden_states, output_hidden_states=True)
                                                                        # hidden_states:
                                                                        #   [bs, L, N, d_hidden]
            pooled_output = vision_outputs.pooler_output                # [bs, d_hidden]
            image_embeds = self.visual_projection(pooled_output)        # [bs, d_out]
            image_embeds = image_embeds.unsqueeze(1)                    # [bs, 1, d_out]

            # NOTE:
            # l2-normalization is occurred when doing cosine loss similarity with text embeds.
            # See transformers.CLIPModel

            out = vision_outputs.hidden_states[split_idx] if split_idx != -1 else image_embeds
            return out, vision_outputs.hidden_states

        self.forward = MethodType(forward, self)

    def _split(self, split_point: str) -> None:
        _client_split_dinov2(self, split_point)

    @overload
    def split(
        self,
        split_point: str,
        keep: bool,
        output_server_model: Literal[False]
    ) -> nn.Module:
        ...

    @overload
    def split(
        self,
        split_point: str,
        keep: bool,
        output_server_model: Literal[True]
    ) -> Tuple[nn.Module, nn.Module]:
        ...

    def split(
        self,
        split_point: str,
        keep: bool = False,
        output_server_model = False
    ) -> nn.Module | Tuple[nn.Module, nn.Module]:
        """ Split model at the given split point.

        Arguments
        ---------
        split_point : str
            Split point.

        keep : bool
            Keep the layers after the split point.
            Default is False to save memory and computation.

        tail : bool
            Construct a nn.Module `tail` to keep the layers after the split point.
            Default is False to save memory and computation.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If split point is invalid.

        NotImplementedError
            If both `keep` and `tail` are specified.
        """
        if bool(keep) and bool(output_server_model):
            raise NotImplementedError("Cannot specify both 'keep' and 'tail'.")

        try:
            if getattr(self, '_splitted') is True:
                raise RuntimeError("Model is already splitted.")
        except AttributeError:
            pass

        server = _server_split_dinov2(
            deepcopy(self), split_point
        ) if output_server_model else None

        if keep:
            self._split_keep(split_point)
        else:
            self._split(split_point)

        self._splitted = True
        return (self, server) if output_server_model else self
