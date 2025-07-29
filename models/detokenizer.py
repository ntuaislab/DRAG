"""
detokenizer.py

Outline
-------


Reference
----------
[1] Darcet, T., Oquab, M., Mairal, J., & Bojanowski, P. (2023). Vision
    transformers need registers. arXiv preprint arXiv:2309.16588.
"""
from dataclasses import asdict, dataclass
from functools import singledispatch
from typing import Dict

import torch
from torch import Tensor, nn
from transformers import CLIPVisionModelWithProjection, Dinov2Model
from transformers.models.clip.configuration_clip import CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPEncoderLayer


def transformer_patchify(tensor: Tensor, patch_size=14) -> Tensor:
    batch_size, channels, height, width = tensor.shape
    assert height % patch_size == 0 and width % patch_size == 0, \
        "Height and width must be divisible by patch_size"

    # Unfold the tensor into patches
    patches = tensor.unfold(2, patch_size, patch_size) \
                    .unfold(3, patch_size, patch_size)

    # Move the patches dimensions to the beginning and reshape
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(batch_size, -1, patch_size * patch_size * channels)

    return patches


def transformer_depatchify(patches: Tensor, patch_size=14) -> Tensor:
    bs, num_patches, token_dim = patches.shape
    # Calculate the number of patches along each dimension
    num_patches_per_dim = int(num_patches ** 0.5)
    channels = token_dim // (patch_size * patch_size)

    # Reshape the patches to the correct dimensions
    patches = patches.view(bs, num_patches_per_dim, num_patches_per_dim, channels, patch_size, patch_size)
    # Permute to bring the channels back to the correct position
    patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
    # Reshape to get the final image
    images = patches.view(bs, channels, num_patches_per_dim * patch_size, num_patches_per_dim * patch_size)

    return images


@dataclass
class TokenizationConfig:
    patch_size: int
    image_size: int
    embed_dim: int

    def __iter__(self):
        return iter((self.patch_size, self.image_size, self.embed_dim))

    def asdict(self):
        return asdict(self)


@singledispatch
def get_tokenization_config(model: nn.Module) -> TokenizationConfig:
    """ Get the tokenization configuration of the VisionTransformer model.

    Parameters
    ----------
    model : nn.Module
        VisionTransformer model.

    Returns
    -------
    Dict[str, Any]
        Tokenization configuration.
    """
    raise NotImplementedError(f"Unsupported model type: {type(model)}")


@get_tokenization_config.register
def _get_clip_vit_tokenization_config(model: CLIPVisionModelWithProjection) -> TokenizationConfig:
    return TokenizationConfig(
        patch_size=model.vision_model.embeddings.patch_size,
        image_size=model.vision_model.embeddings.image_size,
        embed_dim=model.vision_model.embeddings.patch_embedding.out_channels,
    )


@get_tokenization_config.register
def _get_dino_v2_tokenization_config(model: Dinov2Model) -> TokenizationConfig:
    return TokenizationConfig(
        patch_size=model.embeddings.patch_size,
        image_size=224, # DINOv2 accepts dynamic image size, we use 224 to align with CLIP
        embed_dim=model.embeddings.config.hidden_size,
    )


class SRCNN(nn.Module):
    """
    Super resolution convolutional neural network.

    Reference
    ---------
    [1] Dong, C., Loy, C. C., He, K., & Tang, X. (2015). Image super-resolution using deep
        convolutional networks. IEEE transactions on pattern analysis and machine intelligence,
        38(2), 295-307.
    """
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.conv1.weight, mean=0, std=0.001)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.001)
        nn.init.normal_(self.conv3.weight, mean=0, std=0.001)

        nn.init.constant_(self.conv1.bias, 0)
        nn.init.constant_(self.conv2.bias, 0)
        nn.init.constant_(self.conv3.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class Detokenizer(nn.Module):
    """
    Reference
    ---------
    [1] Darcet, T., Oquab, M., Mairal, J., & Bojanowski, P. (2023). Vision transformers
        need registers. arXiv preprint arXiv:2309.16588.

    [2] He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). Masked
        autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference
        on computer vision and pattern recognition (pp. 16000-16009).
    """
    def __init__(
        self,
        embed_dim: int,
        patch_size: int,
        image_size: int,
        decoder_embed_dim: int,
        num_layers: int = 1,
        use_postprocessor: bool = True,
    ) -> None:
        """ Initialize the PatchReconstructor module.

        Parameters
        ----------
        embed_dim: int
            The embedding dimension of the patch tokens.

        patch_size: int
            The size of the patches. Assume square patches (width = height).

        image_size: int
            The size of the image. Assume square images (width = height).

        decoder_embed_dim: int
            The embedding dimension of the decoder.

        num_layers: int
            The number of transformer layers to use.

        use_postprocessor: bool
            Whether to use the SRCNN post-processor.
        """
        super().__init__()

        if num_layers < 0:
            raise ValueError(f"The number of layers must be a positive integer. Recv: {num_layers}")

        num_patches = (image_size // patch_size) ** 2

        config = CLIPVisionConfig(
            hidden_size=decoder_embed_dim,
            num_attention_heads=decoder_embed_dim // 64,    # 64 dim per head
            layer_norm_eps=0.00001,                         # follows default value
        )
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.transformer_layers = nn.ModuleList([
            CLIPEncoderLayer(config=config) for _ in range(num_layers) # type: ignore
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(in_features=decoder_embed_dim, out_features=patch_size * patch_size * 3)
        self.position_embedding = nn.Parameter(torch.randn(num_patches + 1, decoder_embed_dim))
        self.post_processor = SRCNN() if use_postprocessor else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(1, embed_dim))

        self._patch_size = patch_size
        self._image_size = image_size

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def config(self) -> Dict:
        return {
            "_class_name": "PatchReconstructor",
            "embed_dim": self.decoder_embed.in_features,
            "decoder_embed_dim": self.decoder_embed.out_features,
            "patch_size": self._patch_size,
            "image_size": self._image_size,
            "num_layers": len(self.transformer_layers),
            "use_postprocessor": not isinstance(self.post_processor, nn.Identity),
        }

    def mask_tokens(self, tokens: Tensor, mask: Tensor) -> Tensor:
        tokens = tokens.clone()
        tokens[mask] = self.mask_token
        return tokens

    def random_mask(self, tokens: Tensor, mask_ratio: float = 0.0) -> Tensor:
        if not mask_ratio:
            return tokens.clone()

        B, N, _ = tokens.shape  # Batch size, number of tokens, embedding dimension

        # Determine the number of tokens to mask
        num_to_mask = int(mask_ratio * (N-1))

        # Generate random scores for masking
        rand_scores = torch.rand((B, N), device=tokens.device)  # Shape (B, N)
        rand_scores[:, 0] = 1.0  # Always keep the [CLS] token

        # Get the top-k scores to mask
        threshold, _ = torch.topk(rand_scores, num_to_mask, dim=1, largest=False)
        threshold = threshold[:, -1].unsqueeze(1)
        mask = rand_scores <= threshold

        return self.mask_tokens(tokens, mask)

    def forward(self, tokens: Tensor) -> Tensor:
        tokens = self.decoder_embed(tokens) + self.position_embedding
        for layers in self.transformer_layers:
            tokens, = layers(tokens, None, None) # attention_mask=None, causal_attention_mask=None

        tokens = self.decoder_norm(tokens)[:, 1:]
        img_pred = transformer_depatchify(self.decoder_pred(tokens), patch_size=self._patch_size)
        img_pred = self.post_processor(img_pred)

        return img_pred
