"""
split_clip.py

Impl class for splitting transformers.CLIPVisionModelWithProjection.
"""

from copy import deepcopy
from typing import Literal, Tuple, overload

from torch import FloatTensor, Tensor, nn
from transformers import CLIPVisionModelWithProjection
from transformers.models.clip.modeling_clip import CLIPVisionModelOutput

from .base import Splittable


def _client_tokenization_forward(
    self: CLIPVisionModelWithProjection,
    pixel_values: Tensor
) -> Tensor:
    hidden_states = self.vision_model.embeddings(pixel_values)
    hidden_states = self.vision_model.pre_layrnorm(hidden_states)
    return hidden_states


def _client_intermediate_repr_forward(
    self: CLIPVisionModelWithProjection,
    pixel_values: Tensor
) -> Tensor:
    return self.vision_model(pixel_values).last_hidden_state


def _client_embedding_forward(
    self: CLIPVisionModelWithProjection,
    pixel_values: Tensor
) -> Tensor:
    # NOTE:
    # l2-normalization is occurred when doing cosine loss similarity with text embeds.
    # See transformers.CLIPModel
    image_embeds: Tensor

    vision_outputs = self.vision_model(pixel_values=pixel_values)   # hidden_states:
                                                                    #   [bs, N, d_hidden]
    pooled_output = vision_outputs[1]                               # pooled_output:
                                                                    #   [bs, d_hidden]
    image_embeds = self.visual_projection(pooled_output)            # [bs, d_out]
    image_embeds = image_embeds.unsqueeze(1)                        # [bs, 1, d_out]
    return image_embeds


def _client_split_clip_vision_model_with_projection(
    model: CLIPVisionModelWithProjection,
    split_point: str
) -> CLIPVisionModelWithProjection:
    """ Remove the layers after the split point.

    Furthermore, the forward method is replaced with the corresponding method.
    """
    if split_point == 'embeddings':
        # 'embeddings':
        # convert the pixels as tokens, then normalized with LayerNorm

        # Stub out all layers except embedding
        # model.vision_model.pre_layrnorm = nn.Identity()
        model.vision_model.encoder = nn.Identity() # type: ignore
        model.vision_model.post_layernorm = nn.Identity() # type: ignore
        model.visual_projection = nn.Identity() # type: ignore
        model.forward = _client_tokenization_forward.__get__( # pylint: disable=no-value-for-parameter
            model, type(model)
        )

        return model

    if split_point.startswith('encoder_layer'):
        # 'encoder_layer_<n>':
        # convert the pixels as tokens, and pass through the first n layers of the encoder

        split_idx = int(split_point.split('_')[-1])
        if split_idx > len(model.vision_model.encoder.layers):
            raise ValueError(f"Invalid split point: {split_point}")

        model.vision_model.encoder.layers = nn.ModuleList(
            model.vision_model.encoder.layers[:split_idx] # type: ignore
        )
        # Stub out all remaining layers
        model.vision_model.post_layernorm = nn.Identity() # type: ignore
        model.visual_projection = nn.Identity() # type: ignore
        model.forward = _client_intermediate_repr_forward.__get__( # pylint: disable=no-value-for-parameter
            model, type(model)
        )

        return model

    if split_point == 'image_embeds':
        # 'image_embeds':
        # pass through the entire model except the normalization
        model.forward = _client_embedding_forward.__get__( # pylint: disable=no-value-for-parameter
            model, type(model)
        )

        return model

    raise ValueError(f"Invalid split point: {split_point}")


def _server_intermediate_repr_forward(
    self: CLIPVisionModelWithProjection,
    intermediate_repr: Tensor
) -> CLIPVisionModelOutput:
    # NOTE:
    # l2-normalization is occurred when doing cosine loss similarity with text embeds.
    # See transformers.CLIPModel
    image_embeds: FloatTensor

    # hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
    # hidden_states = self.pre_layrnorm(hidden_states)

    encoder_outputs = self.vision_model.encoder(                            # hidden_states:
        inputs_embeds=intermediate_repr,                                    #   [bs, N, d_hidden]
    )

    last_hidden_state = encoder_outputs[0]
    pooled_output = last_hidden_state[:, 0, :]                              # pooled_output:
    pooled_output = self.vision_model.post_layernorm(pooled_output)         #   [bs, d_hidden]
    image_embeds = self.visual_projection(pooled_output)                    # [bs, d_out]
    return CLIPVisionModelOutput(
        image_embeds=image_embeds,                                          # [bs, d_out]
    )


def _server_split_clip_vision_model_with_projection(
    model: CLIPVisionModelWithProjection,
    split_point: str
) -> CLIPVisionModelWithProjection:
    if split_point == 'embeddings':
        raise NotImplementedError

    if split_point.startswith('encoder_layer'):
        # 'encoder_layer_<n>':
        # convert the pixels as tokens, and pass through the first n layers of the encoder

        split_idx = int(split_point.split('_')[-1])
        if split_idx > len(model.vision_model.encoder.layers):
            raise ValueError(f"Invalid split point: {split_point}")

        model.vision_model.embeddings = nn.Identity() # type: ignore
        model.vision_model.pre_layrnorm = nn.Identity() # type: ignore
        model.vision_model.encoder.layers = nn.ModuleList(
            model.vision_model.encoder.layers[split_idx:] # type: ignore
        )

        model.forward = _server_intermediate_repr_forward.__get__( # pylint: disable=no-value-for-parameter
            model, type(model)
        )

        return model

    if split_point == 'image_embeds':
        return nn.Identity() # type: ignore

    raise ValueError(f"Invalid split point: {split_point}")


class SplittableCLIP(CLIPVisionModelWithProjection, Splittable):
    """ Run CLIPVisionModelWithProjection with collaborative inference mode. """
    def _split_keep(self, split_point: str) -> None:
        # pylint: disable=invalid-name
        match split_point:
            case 'embeddings':
                split_idx = 0
            case 'image_embeds':
                split_idx = -1
            case _ if split_point.startswith('encoder_layer'):
                split_idx = int(split_point.split('_')[-1])
                if split_idx > len(self.vision_model.encoder.layers):
                    raise ValueError(f"Invalid split point: {split_point}")
            case _:
                raise ValueError(f"Invalid split point: {split_point}")

        def forward(self, pixel_values: Tensor) -> Tuple[Tensor, Tuple[Tensor]]:
            vision_outputs = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=True
            )                                                           # hidden_states:
                                                                        #   [bs, L, N, d_hidden]
            pooled_output = vision_outputs.pooler_output                # [bs, d_hidden]
            image_embeds = self.visual_projection(pooled_output)        # [bs, d_out]
            image_embeds = image_embeds.unsqueeze(1)                    # [bs, 1, d_out]

            # NOTE:
            # l2-normalization is occurred when doing cosine loss similarity with text embeds.
            # See transformers.CLIPModel

            out = vision_outputs.hidden_states[split_idx] if split_idx != -1 else image_embeds
            return out, vision_outputs.hidden_states

        self.forward = forward.__get__(self, SplittableCLIP) # pylint: disable=no-value-for-parameter

    def _split(self, split_point: str) -> None:
        _client_split_clip_vision_model_with_projection(self, split_point)

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

        server = _server_split_clip_vision_model_with_projection(
            deepcopy(self), split_point
        ) if output_server_model else None

        if keep:
            self._split_keep(split_point)
        else:
            self._split(split_point)

        self._splitted = True
        return (self, server) if output_server_model else self
