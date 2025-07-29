"""
Paper: GAN You See Me? Enhanced Data Reconstruction Attacks against Split Inference
https://openreview.net/forum?id=YZGWhs1H7F

Applies StyleGAN-XL as the manifold restriction.
"""

# pylint: disable=wrong-import-position
from enum import Enum
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import TotalVariation  # pylint: disable=import-error
from torchvision.transforms import v2
from tqdm import tqdm

from models import use_criterion
from models.distance import dino_image_similarity
from models.stylegan import load_model, redefine_mapping_network
from runner.base import TEST
from runner.data_reconstruction import (OptimizationBasedDRA,
                                        ReconstructionMetric)
from runner.utils import create_optimizer, save_image, save_tensor


class ClassEmbeddingType(Enum):
    """ Enum class for class embedding type. """
    EMBEDDING = 'embedding'
    LOGITS = 'logits'
    NONE = None

    @classmethod
    def from_string(cls, x: str) -> 'ClassEmbeddingType':
        try:
            return cls(x)
        except ValueError:
            raise ValueError(f'Unsupported type to represent condition in StyleGAN: {x}')


class GLASS(OptimizationBasedDRA):
    """ White-box attack via searching in the latent space of StyleGAN2. """

    generator: nn.Module

    distance_fn: nn.Module

    @property
    def name(self) -> str:
        return 'glass'

    writer: SummaryWriter
    device: torch.device

    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        super().__init__(config=config)

        self.writer = SummaryWriter(log_dir=Path.cwd())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _prepare_model(self, **kwargs) -> None:
        super()._prepare_model(**kwargs)

        cfg = self.configs
        ckpt = self.configs.image_prior.checkpoint
        device = self.device

        generator = load_model(ckpt).to(device)
        generator.mapping = redefine_mapping_network(generator.mapping)

        self.generator = generator
        self.distance_fn = use_criterion(cfg.distance_fn)

    def get_class_embedding(self, c: Tensor | None, repr_type: ClassEmbeddingType) -> Tensor | None:
        if repr_type == ClassEmbeddingType.NONE:
            return None

        assert c is not None
        if repr_type == ClassEmbeddingType.EMBEDDING:
            return c

        if repr_type == ClassEmbeddingType.LOGITS:
            c = c.softmax(dim=-1)

        return c @ self.generator.mapping.embed.weight

    def run(self) -> None:
        # Configuration
        cfg = self.configs
        device = self.device

        self._prepare_model()
        self._prepare_dataset()
        self.criteria = ReconstructionMetric(**dino_image_similarity()).to(device)

        generator, client_model = self.generator, self.client_model

        # Training index
        tv = TotalVariation(reduction='mean').to(device)

        # Hyperparameters
        psi = cfg.image_prior.truncation_psi
        lamda = cfg.objective.kl_divergence
        alpha = cfg.objective.total_variation

        res = cfg.model.image_size
        client_model.eval()

        (intermediate_repr, mask), x_target = self.dataset[TEST]

        cls_embeds_type = ClassEmbeddingType.from_string(cfg.image_prior.class_embedding)

        if cls_embeds_type == ClassEmbeddingType.EMBEDDING:
            c = nn.Parameter(generator.mapping.embed.weight.mean(0, keepdim=True).to(device))
        elif cls_embeds_type == ClassEmbeddingType.LOGITS:
            c = nn.Parameter(torch.zeros(1, generator.c_dim, device=device))
        elif cls_embeds_type == ClassEmbeddingType.NONE:
            c = None
        else:
            raise ValueError(f"Unknown type of class embedding: {cls_embeds_type}")

        # Z ~ N(0, 1)
        z = nn.Parameter(torch.randn(1, generator.z_dim, device=device))

        optimizer_config = OmegaConf.to_container(cfg.optimizer, resolve=True)
        assert isinstance(optimizer_config, dict)

        optimizer = create_optimizer(
            optimizer_config.pop('_target_'),
            [z] if c is None else [z, c],
            **optimizer_config,
        )

        pbar = tqdm(range(1, cfg.max_steps + 1), desc='Z-Space', ncols=0)
        for num_iter in pbar:
            # z -> w -> x_reconstruct
            e = self.get_class_embedding(c, cls_embeds_type)
            w_plus = generator.mapping(z, e, truncation_psi=psi)
            im = generator.synthesis(w_plus, noise_mode='const', force_fp32=True) # NCHW, float32
                                                                                  # [-1, +1]
            im = im * 0.5 + 0.5
            im = v2.Resize(res, antialias=True)(im)
            im.clamp_(0., 1.)

            # embed reconstruct
            dH = self.distance_fn(
                client_model(self.client_preprocessor(im)).squeeze(0)[mask],
                intermediate_repr.squeeze(0)[mask]
            )
            tv_reg = tv(im) / (3 * res * res)
            tv.reset()
            kl_reg = -0.5 * (
                1 + z.std(dim=1).pow(2).log() - z.mean(dim=1).pow(2) - z.std(dim=1).pow(2)
            ).sum()
            loss = dH + lamda * kl_reg + alpha * tv_reg

            # Optimization step.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            if num_iter % cfg.val_check_interval == 0:
                metrics = self.criteria(im, x_target)
                self.criteria.reset()
                for k, v in metrics.items():
                    self.writer.add_scalar(f'Metric/{k}', v, num_iter)

                assert z.grad is not None
                norm = z.grad.norm(dim=1).mean()
                self.writer.add_images('Reconstruction', im, num_iter)
                self.writer.add_scalar('Training/Gradient Norm', norm.item(), num_iter)
                self.writer.add_scalar('Training/Total Variation', tv_reg.item(), num_iter)
                self.writer.add_scalar('Training/Hidden State Distance', dH.item(), num_iter)
                self.writer.add_scalar('Training/KL Divergence', kl_reg.item(), num_iter)

        save_tensor(z, f'z.pt')
        save_image(im, f'image_{cfg.max_steps:06d}')

        with torch.no_grad():
            e = self.get_class_embedding(c, cls_embeds_type)
            w_plus = generator.mapping(z, e, truncation_psi=psi)

        w_plus = nn.Parameter(w_plus)
        optimizer_config = OmegaConf.to_container(cfg.optimizer, resolve=True)
        assert isinstance(optimizer_config, dict)

        optimizer = create_optimizer(optimizer_config.pop('_target_'), [w_plus], **optimizer_config)

        pbar = tqdm(range(cfg.max_steps + 1, 2 * cfg.max_steps + 1), desc='W-Space', ncols=0)
        for num_iter in pbar:
            im = generator.synthesis(w_plus, noise_mode='const', force_fp32=True) # NCHW, float32, dynamic range [-1, +1]
            im = im * 0.5 + 0.5
            im = v2.Resize(res, antialias=True)(im)
            im.clamp_(0., 1.)

            dH = self.distance_fn(
                client_model(self.client_preprocessor(im)).squeeze(0),
                intermediate_repr.squeeze(0)
            )
            tv_reg = tv(im) / (3 * res * res)
            loss = dH + alpha * tv_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if num_iter % cfg.val_check_interval == 0:
                metrics = self.criteria(im, x_target)
                self.criteria.reset()
                for k, v in metrics.items():
                    self.writer.add_scalar(f'Metric/{k}', v, num_iter)

                assert w_plus.grad is not None
                norm = w_plus.grad.norm(dim=1).mean()
                self.writer.add_images('Reconstruction', im, num_iter)
                self.writer.add_scalar('Training/Gradient Norm', norm.item(), num_iter)
                self.writer.add_scalar('Training/Total Variation', tv_reg.item(), num_iter)
                self.writer.add_scalar('Training/Hidden State Distance', dH.item(), num_iter)
                self.writer.add_scalar('Training/KL Divergence', torch.tensor(float('nan')), num_iter)

        save_tensor(w_plus, f'w.pt')
        save_image(im, f'image_{2*cfg.max_steps:06d}')

        metrics = self.criteria(im, x_target)
        self.criteria.reset()
        self.writer.add_hparams(self.hparams, metrics)


@hydra.main(config_path="config",
            config_name=Path(__file__).stem,
            version_base='1.1')
def main(config: DictConfig) -> None: # pylint: disable=missing-function-docstring
    task = GLASS(config=config)
    task.run()

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
