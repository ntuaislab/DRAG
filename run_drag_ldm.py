"""
run_drag_ldm.py

Optimize latent z_t to reconstruct an image x, given the encoder
`client_model` and its output `intermediate_repr`.
"""

# pylint: disable=wrong-import-position
import gc
from pathlib import Path
from typing import Callable

import hydra
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import \
    StableDiffusionImg2ImgPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from omegaconf import DictConfig, OmegaConf
from torch import FloatTensor, Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional.image.tv import \
    _total_variation_update as total_variation  # pylint: disable=import-error
from torchvision.transforms import v2
from tqdm import tqdm

from models import use_criterion
from models.detokenizer import Detokenizer
from models.diffusion import Sample
from models.distance import dino_image_similarity
from models.functional import slerp
from models.optimize_utils import get_step
from models.stable_diffusion import get_unconditioned_embedding
from runner.base import TEST
from runner.data_reconstruction import (OptimizationBasedDRA,
                                        ReconstructionMetric)
from runner.dataset import LDM_IMG_TRANSFORM
from runner.utils import create_optimizer, parse_torch_dtype, save_image


class DRAG(OptimizationBasedDRA):
    """ Optimization-based data reconstruction via DRAG: Data Reconstruction
    Attack with Guided Diffusion.

    Using latent diffusion model as the image prior.
    """

    vae: AutoencoderKL
    unet: UNet2DConditionModel
    pipe: StableDiffusionImg2ImgPipeline
    noise_scheduler: DDIMScheduler | SchedulerMixin | None

    distance_fn: nn.Module
    torch_dtype: torch.dtype
    unconditioned_embedding: FloatTensor

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
        dtype = parse_torch_dtype(self.configs.image_prior.torch_dtype)
        device = self.device

        pipe: StableDiffusionImg2ImgPipeline
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            ckpt, torch_dtype=dtype, safety_checker=None,
        ).to(device)
        pipe.set_progress_bar_config(disable=True)
        vae = pipe.vae

        unet = pipe.unet
        noise_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        assert isinstance(noise_scheduler, DDIMScheduler)
        noise_scheduler.register_to_config(steps_offset=0)

        self.logger.info('Generator: %s(%s) (dtype: %s)',
                         cfg.image_prior.name, cfg.image_prior.checkpoint, dtype)

        self.vae = vae
        self.unet = unet
        self.pipe = pipe
        self.noise_scheduler = noise_scheduler

        self.distance_fn = use_criterion(cfg.distance_fn)
        self.torch_dtype = dtype

    @torch.no_grad()
    def _inversion(self, z: FloatTensor, strength: float = 1.0) -> Tensor:
        cfg = self.configs
        num_inference_steps = self.configs.image_prior.generate_kwargs.num_inference_steps

        device = z.device

        scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config)
        assert isinstance(scheduler, DDPMScheduler)
        scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)
        timesteps, num_inference_steps = scheduler.timesteps, num_inference_steps
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps, cfg.image_prior.generate_kwargs.strength, device=device)

        noise = torch.randn_like(z)
        return scheduler.add_noise(z, noise, timesteps[:1]) # type: ignore

    @torch.no_grad()
    def _init_from_reconstructor(
        self,
        ckpt_path: str | Path,
        intermediate_repr: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        ckpt_path = Path(ckpt_path)
        state_dict = torch.load(ckpt_path / 'last.ckpt', map_location='cpu')['state_dict']
        with open(ckpt_path / 'model_index.json', 'r') as f:
            model_config = OmegaConf.load(f)
            assert isinstance(model_config, DictConfig)

        match (class_name := model_config.pop('_class_name')):
            case 'PatchReconstructor':
                model = Detokenizer(**model_config) # type: ignore
                model.load_state_dict({
                    k.replace('reconstructor.', ''): v
                        for k, v in state_dict.items()
                            if k.startswith('reconstructor.')
                })
                model = model.to(self.device)

                assert mask is not None
                x = model(model.mask_tokens(intermediate_repr, ~mask.unsqueeze(0)))

            case _:
                raise ValueError(f'Unsupported reconstructor: {class_name}')

        x = self.client_unnormalizer(x)
        x = v2.functional.resize(x, [512, 512], antialias=True).clamp(0, 1)
        return LDM_IMG_TRANSFORM(x) # [-1, 1]

    @torch.no_grad()
    def initialize_latent(self, intermediate_repr: Tensor, mask: Tensor) -> Tensor:
        x: Tensor # Image represented in range [-1, 1]
        z: Tensor # Latent code

        # Helper variables
        cfg = self.configs
        strength = cfg.image_prior.generate_kwargs.strength

        latent_config = cfg.image_prior.latent
        assert isinstance(latent_config, DictConfig)

        init_from = latent_config.init._from_
        if init_from is None:
            shape = OmegaConf.to_object(self.configs.image_prior.latent.shape)
            assert isinstance(shape, list) and len(shape) == 4

            z = torch.randn(size=shape, device=self.device, dtype=self.torch_dtype)
            self.logger.info('Initialized latent code from N(0, 1)')
            return z

        x = self._init_from_reconstructor(init_from, intermediate_repr, mask)
        self.logger.info('Initialized latent code from %s', init_from)

        self.writer.add_images('One-step reconstruction', ((x + 1) / 2).to(torch.float32), 0)
        z = self.vae.encode(x).latent_dist.mode() * self.vae.config.scaling_factor  # type: ignore
        return self._inversion(z, strength)

    def get_timesteps(self, num_inference_steps: int, strength: float, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.noise_scheduler.timesteps[t_start * self.noise_scheduler.order :] # type: ignore
        if hasattr(self.noise_scheduler, "set_begin_index"):
            self.noise_scheduler.set_begin_index(t_start * self.noise_scheduler.order) # type: ignore

        return timesteps, num_inference_steps - t_start

    def regularization(self, x: Tensor) -> Tensor:
        """ Regularization term for the optimization-based DRA task.

        Parameters
        ----------
        x : Tensor
            The input tensor. Shape: [B, C, H, W]. Range: [-1, 1].
        """
        l2_reg = x.pow(2).mean()
        tv_reg, _ = total_variation((x + 1) / 2)
        tv_reg = tv_reg.mean()     # Measure total variation loss on [0, 1]
        return (
            self.configs.regularization.total_variation_x * tv_reg
            + self.configs.regularization.l2_regularization_x * l2_reg
        )

    @torch.no_grad()
    def run(self):
        self._prepare_model()
        self._prepare_dataset()

        # Helper variables
        cfg = self.configs
        vae = self.vae
        device = self.device
        scheduler = self.noise_scheduler
        client_model = self.client_model

        assert scheduler.config.prediction_type == "epsilon"  # type: disable: no-member

        self.criteria = ReconstructionMetric(**dino_image_similarity()).to(device)

        (intermediate_repr, mask), x_target = self.dataset[TEST]
        # x_target = x_target.unsqueeze(0)

        self.unconditioned_embedding = get_unconditioned_embedding(
            self.pipe.text_encoder,
            self.pipe.tokenizer,
            normalize=True,
        ) # type: ignore

        k = cfg.image_prior.generate_kwargs.num_internal_iterations
        g_r = cfg.image_prior.generate_kwargs.guidance_rate
        eta = cfg.image_prior.generate_kwargs.eta
        max_grad_norm = float(cfg.image_prior.max_grad_norm)
        num_inference_steps = cfg.image_prior.generate_kwargs.num_inference_steps

        assert isinstance(k, int) and k > 0

        # Adapted from StableDiffusionImg2ImgPipeline, retrieve the denoising schedule
        scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)  # type: ignore
        timesteps, num_inference_steps = scheduler.timesteps, num_inference_steps   # type: ignore
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps, cfg.image_prior.generate_kwargs.strength, device=device)

        z_t = self.initialize_latent(intermediate_repr, mask)
        self.logger.info('Initialized latent code with shape: %s', z_t.shape)

        interp_fn: Callable
        match cfg.image_prior.generate_kwargs.interpolation:
            case 'lerp':
                interp_fn = torch.lerp

            case 'slerp':
                interp_fn = slerp

            case _:
                raise ValueError(f'Unsupported interpolation method: {cfg.image_prior.generate_kwargs.interpolation}')

        z_t = Sample(z_t.clone().detach(), timesteps[0], scheduler)                 # type: ignore
        optimizer_config = OmegaConf.to_container(cfg.optimizer, resolve=True)
        assert isinstance(optimizer_config, dict)
        optimizer = create_optimizer(
            optimizer_config.pop('_target_'),
            z_t.parameters(),
            lr=0,
            **optimizer_config, # type: ignore
        )

        x_t = vae.decode(z_t.unwrap() / vae.config.scaling_factor).sample  # [-1, 1] # type: ignore
        x_t = v2.functional.resize(x_t, [224, 224])                        # [-1, 1]
        x_t = ((x_t + 1) / 2).clamp(0, 1)                                  # [ 0, 1]

        pbar = tqdm(range(k * num_inference_steps), desc='Optimizing', ncols=0)
        for i, t in enumerate(z_t.timesteps, 1):
            for n in range(1, k + 1):
                # Calculate ∇x_t log p(y|x)
                with torch.enable_grad():
                    pred_epsilon = self.unet(
                        sample=z_t.sample,
                        timestep=t,
                        encoder_hidden_states=self.unconditioned_embedding
                    ).sample

                    z0_est = z_t.tweedie(pred_epsilon)
                    x0_est = vae.decode(z0_est / vae.config.scaling_factor).sample  # [-1, 1] # type: ignore
                    x0_est = v2.functional.resize(x0_est, [224, 224])               # [-1, 1]

                    dH: Tensor = self.distance_fn(
                        client_model(self.client_preprocessor(
                            ((x0_est + 1) / 2).clamp(0, 1)
                        )).squeeze(0)[mask],
                        intermediate_repr.squeeze(0)[mask]
                    )
                    if dH.isnan().any():
                        torch.save(x0_est, 'x0_est.pt')
                        raise ValueError('NaN detected in the distance function!')

                    optimizer.zero_grad()                                           # g := 0
                    loss = dH + self.regularization(x0_est)
                    loss.backward()                                                 # g := 0 + ∇x_t

                grad_norm = clip_grad_norm_(z_t.unwrap(), max_grad_norm)
                optimizer.step()                                                    # Calculate m_t and v_t (if necessary)
                step = get_step(z_t.sample, optimizer, clone=False)                 # Extract s := ∇x_t log p(y|x) (or its substitute)

                # Sample under Spherical Gaussian constraint
                std = z_t.std_dev_t(eta=eta)
                r = (z_t.sample.numel() ** 0.5) * std
                v = torch.nn.functional.normalize(step, p=2, dim=(1, 2, 3))     # type: ignore
                d_star = -r * v                                                 # d* = -r * ∇x_t / ||∇x_t||
                d_sample = std * torch.randn_like(z_t.sample)                   # d_sample = σ_t * ε
                d = interp_fn(d_sample, d_star, g_r)
                d = r * torch.nn.functional.normalize(d, p=2, dim=(1, 2, 3))    # d = r * d_m / ||d_m|| # type: ignore

                # Sampling x_{t-1}
                z_t = z_t.denoise(pred_epsilon, noise=d, eta=eta)

                # Self-Recurrence: Eq. (10)
                if n != k:
                    z_t = z_t.diffuse(step=1)

                # Logging to progress bar
                pbar.set_postfix({ 'distance': dH.item() })
                pbar.update()

            x0_est = ((x0_est + 1) / 2).clamp(0, 1)                                # [ 0, 1] # type: ignore

            # Logging to TensorBoard
            training_log = {
                'Distance': dH.item(),                      # type: ignore
                'GradNorm': grad_norm.item(),               # type: ignore
                'Timestep': t
            }

            for key, value in training_log.items():
                self.writer.add_scalar(f'Training/{key}',
                                       value,
                                       i * cfg.image_prior.generate_kwargs.num_internal_iterations)

            if i % cfg.log_every_n_steps == 0:
                self.writer.add_images('One-step reconstruction',
                                        x0_est.to(torch.float32),
                                        i * cfg.image_prior.generate_kwargs.num_internal_iterations)

            metrics = self.criteria(x0_est, x_target)
            self.criteria.reset()
            for key, value in metrics.items(): # pylint: disable=invalid-name
                self.writer.add_scalar(f'Metric/{key}', value, i * cfg.image_prior.generate_kwargs.num_internal_iterations)

        # Final Log
        x0 = vae.decode(z_t.unwrap() / vae.config.scaling_factor).sample           # [-1, 1] # type: ignore
        x0 = v2.functional.resize((x0 + 1) / 2, [224, 224]).clamp(0, 1)            # [ 0, 1]
        self.writer.add_images('Reconstruction',
                               x0.to(torch.float32),
                               len(z_t.timesteps) * k)
        save_image(x0, 'image')

        metrics = self.criteria(x0, x_target)
        self.criteria.reset()
        self.writer.add_hparams(self.hparams, metrics)

        self.logger.info('Optimization completed!')
        self.logger.info(metrics)


@hydra.main(config_path="config",
            config_name=Path(__file__).stem,
            version_base='1.1')
def main(config: DictConfig) -> None:
    # pylint: disable=missing-function-docstring

    def run():
        task = DRAG(config=config)
        task.run()

    run()

    # Force free up GPU memory after the execution
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
