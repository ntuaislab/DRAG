"""
clip_classifier.py

This script contains the adapter class.
"""

# pylint: disable=missing-class-docstring, missing-function-docstring
from torch import Tensor, nn
from transformers import CLIPVisionModelWithProjection

# Difference between CLIPVisionModelWithProjection and ModifiedResNet
#
# Accessing classification head
# >>> model.visual_projection            # CLIPVisionModelWithProjection
# >>> model[-1]                          # ModifiedResNet
#
# Model forward
# >>> model(
#   pixel_values=pixel_values,
#   output_attentions=None,
#   output_hidden_states=None,
#   return_dict=None
# ) # CLIPVisionModelWithProjection
# >>> model(x=x)                         # ModifiedResNet


class ClientAdapter(nn.Module):
    def __init__(
        self,
        client: CLIPVisionModelWithProjection | nn.Module,
    ) -> None:
        super().__init__()

        self.client = client

    def forward(self, pixel_values: Tensor) -> Tensor:
        return self.client(pixel_values)


class ServerAdapter(nn.Module):
    def __init__(
        self,
        server: CLIPVisionModelWithProjection | nn.Sequential,
    ) -> None:
        super().__init__()

        self.server = server

    @property
    def head(self) -> nn.Linear:
        if isinstance(self.server, CLIPVisionModelWithProjection):
            return self.server.visual_projection

        head: nn.Linear = self.server[-1] # type: ignore
        return head

    def forward(self, intermediate_repr: Tensor) -> Tensor:
        if isinstance(self.server, CLIPVisionModelWithProjection):
            logits: Tensor = self.server(intermediate_repr).image_embeds

            # Empirically, NOT normalizing either the image or text embeddings does not
            # hurt the performance too much => following the convention of the classic
            # classification model.
            # image_embeds: Tensor = F.normalize(image_embeds, p=2, dim=-1)

            # logits: Tensor = self.head(image_embeds)

            return logits

        logits: Tensor = self.server(intermediate_repr)
        return logits
