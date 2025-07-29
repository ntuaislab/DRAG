"""
distance.py
-----------
Impl of distance and loss functions.
"""

from typing import Callable, Dict

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import CosineSimilarity
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
from torchvision.transforms import v2
from transformers import AutoModel
from transformers import ViTModel as DinoModel


class EmbeddingModel(nn.Module):
    def __init__(self, model: DinoModel):
        super().__init__()

        self.model: DinoModel = model.eval()
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x).pooler_output


def dino_image_similarity() -> Dict:
    # preprocessor for facebook/DINOv2 which retains gradient and device
    DINOV2_IMG_TRANSFORM = v2.Compose([
        v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop(224),
        v2.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    checkpoint = 'facebook/dino-vits16'

    # Warning:
    #
    # Some weights of ViTModel were not initialized from the model checkpoint at facebook/dino-vits16
    # and are newly initialized: ['pooler.dense.bias', 'pooler.dense.weight']. You should probably
    # TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    model: DinoModel = AutoModel.from_pretrained(checkpoint)

    # https://arxiv.org/pdf/2306.09344: DINO, CLS before layernorm
    model.layernorm = nn.Identity()

    return {
        'embedding_model': EmbeddingModel(model),
        'preprocessing': DINOV2_IMG_TRANSFORM,
    }


def patch_prior_distance(img: Tensor, patch_size=14, reduction: str = 'mean') -> Tensor:
    """ Regularize spatial positioning of neighboring patches by enforcing spatial smoothness.

    Arguments
    ---------
    img : torch.Tensor
        Input tensor in shape (N, C, H, W).

    patch_size : int
        ViT patch size.

    reduction : str
        Reduction method. Default: 'mean'.

    Returns
    -------
    Tensor
        Loss value.

    References
    ----------
    .. [1] Hatamizadeh, A., Yin, H., Roth, H. R., Li, W., Kautz, J., Xu, D., & Molchanov, P. (2022).
           GradViT: Gradient inversion of vision transformers. In Proceedings of the IEEE/CVF
           Conference on Computer Vision and Pattern Recognition (pp. 10021-10030).
           https://arxiv.org/pdf/2203.11894.pdf
    """
    assert len(img.size()) == 4, f'Expected 4D tensor, got {img.size()}.'
    assert all(map(lambda d: d % patch_size == 0, img.size()[2:])), \
        f'Patch size ({patch_size}) does not divide input width and height {img.size()[2:]}.'

    rows = torch.arange(patch_size, img.size(2), patch_size, device=img.device)
    cols = torch.arange(patch_size, img.size(3), patch_size, device=img.device)

    # Compute patch prior distance
    rows_diff = img[:, :, rows, :] - img[:, :, rows - 1, :]
    rows_loss = rows_diff.norm(p=2, dim=(1, 3)).sum(dim=-1)

    cols_diff = img[:, :, :, cols] - img[:, :, :, cols - 1]
    cols_loss = cols_diff.norm(p=2, dim=(1, 2)).sum(dim=-1)

    patch_diff = rows_loss + cols_loss

    if reduction == 'none':
        return patch_diff

    if reduction == 'mean':
        return patch_diff.mean()

    if reduction == 'sum':
        return patch_diff.sum()

    raise ValueError(f"Unknown reduction: {reduction}")


def patchify(x: Tensor, patch_size: int, num_channels: int):
    """ Regularize spatial positioning of neighboring patches by enforcing spatial smoothness.

    Arguments
    ---------
    img : torch.Tensor
        Input tensor in shape (N, C, H, W).

    patch_size : int
        ViT patch size.

    reduction : str
        Reduction method. Default: 'mean'.

    Returns
    -------
    Tensor
        Loss value.

    References
    ----------
    .. [1] Hatamizadeh, A., Yin, H., Roth, H. R., Li, W., Kautz, J., Xu, D., & Molchanov, P. (2022).
           GradViT: Gradient inversion of vision transformers. In Proceedings of the IEEE/CVF
           Conference on Computer Vision and Pattern Recognition (pp. 10021-10030).
           https://arxiv.org/pdf/2203.11894.pdf
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)

    n = x.size(0)
    return (
        x.unfold(1, num_channels, num_channels)             # (1, 1, H,   W,   c)
         .unfold(2, patch_size, patch_size)                 # (1, 1, H/f, W,   c, f)
         .unfold(3, patch_size, patch_size)                 # (1, 1, H/f, W/f, c, f, f)
         .contiguous()
         .view(n, -1, num_channels, patch_size, patch_size) # (H/f * W/f, c, f, f)
    )


class CosineSimilarityLoss(CosineSimilarity):
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction: str = reduction

    def forward(self, x1: Tensor, x2: Tensor, reduction=None) -> Tensor:
        """ Forward pass.

        Arguments
        ---------
        x1, x2 : torch.Tensor
            Input and target tensor.

        reduction : str
            Reduction method.
            Follows the setting during construction if not specified.

        Returns
        -------
        Tensor
            Loss value.
        """
        assert x1.dim() in (2, 3) and x2.dim() in (2, 3), \
            f'Expected 2D or 3D tensor, got {x1.size()} and {x2.size()}.'
        reduction = reduction or self.reduction

        # Permute to fulfill the requirements of torch.nn.CosineSimilarity.
        if x1.dim() == 3:
            x1 = x1.permute(0, 2, 1)

        if x2.dim() == 3:
            x2 = x2.permute(0, 2, 1)

        # out in range [0, 2]
        out = 1 - super().forward(x1, x2)

        if reduction == 'none':
            return out

        if reduction == 'elementwise_mean':
            return out.mean(dim=-1)

        if reduction == 'mean':
            return out.mean()

        if reduction == 'sum':
            if len(out.size()) == 2:
                return out.mean(dim=-1).sum()
            return out.sum()

        raise ValueError(f'Unknown reduction: {reduction}')


class AdaptiveCosineSimilarityLoss(CosineSimilarityLoss):
    """ Loss function designed for bypassing DISCO defense.

    Urging the intermediate repr to fit the target one which some
    channels are pruned `misleads` the reconstruction process.

    One way to bypass it is to guess which channels are pruned,
    and we could ignore these pruned channels to avoid misleading.
    """
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__(reduction=reduction)

        self.channel_mask: Tensor | None = None

    def forward(self, x1: Tensor, x2: Tensor, reduction=None) -> Tensor:
        # Design for guessing the channel being masked out by DISCO
        if self.channel_mask is None:
            self.channel_mask = (x2.abs().mean(dim=0) > 0.1).float()

            print(f'Filtered channel ratio: {self.channel_mask.mean()=}')
            assert self.channel_mask.ndim == 1, \
                f'Expected 1D tensor, got {self.channel_mask.size()}.'

        x1 = x1 * self.channel_mask
        x2 = x2 * self.channel_mask
        return super().forward(x1, x2, reduction)


def _pairwise_euclidean_distances(x: Tensor) -> Tensor:
    """
    Taken from: https://discuss.pytorch.org/t/batched-pairwise-distance/39611
    """
    x_norm = x.pow(2).sum(dim=1, keepdim=True).view(-1, 1)

    # Law of cosines
    # => dist is the squared l2 distance
    dist = x_norm + x_norm.T - 2.0 * torch.mm(x, x.T)
    dist[dist != dist] = 0  # replace nan values with 0
    return torch.clamp(dist, 1.0e-6, np.inf).sqrt() # clamp to avoid undefined gradient
    # return torch.clamp(dist, 0.0, np.inf) #.sqrt()


def _pairwise_cosine_distances(x: Tensor) -> Tensor:
    """
    Arguments
    ---------
    x : (bs, n, d) Tensor
        Input tensor.
    """
    _, n, _ = x.size()
    x = F.normalize(x, p=2, dim=-1)
    x = x.flatten(1)
    return 1 - (x @ x.T / n)


def pairwise_euclidean_distances(x: Tensor) -> Tensor:
    return _pairwise_euclidean_distances(x.flatten(1))


def pairwise_cosine_distances(x: Tensor) -> Tensor:
    return _pairwise_cosine_distances(x)


def distance_correlation(
    x: Tensor,
    z: Tensor,
    dx: Callable[[Tensor], Tensor] = pairwise_euclidean_distances,
    dz: Callable[[Tensor], Tensor] = pairwise_euclidean_distances,
) -> Tensor:
    x = dx(x)
    x = x - x.mean(dim=0).unsqueeze(1) - x.mean(dim=1) + x.mean()

    z = dz(z)
    z = z - z.mean(dim=0).unsqueeze(1) - z.mean(dim=1) + z.mean()
    dCOV2ab = (x * z).mean()
    var2_aa = (x * x).mean()
    var2_bb = (z * z).mean()

    dCOR2ab = dCOV2ab / torch.sqrt(var2_aa * var2_bb)
    return dCOR2ab.sqrt()


class DistCorrelation(_Loss):
    def __init__(
        self,
        dx: Callable[[Tensor], Tensor] = pairwise_euclidean_distances,
        dz: Callable[[Tensor], Tensor] = pairwise_euclidean_distances,
    ) -> None:
        super().__init__(reduction=None)

        self.dx = dx
        self.dz = dz

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        return distance_correlation(x, z, self.dx, self.dz)
