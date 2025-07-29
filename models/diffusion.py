"""
diffusion.py
"""

import torch
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from torch import Tensor, nn


class Sample(nn.Module):
    """ Follows denoising diffusion implicit models. """
    final_alpha_cumprod: Tensor
    alphas_cumprod: Tensor

    num_train_timesteps: int
    num_inference_steps: int
    step_size: int

    sample: Tensor
    timestep: Tensor

    def __init__(
        self,
        sample: Tensor,
        timestep: Tensor,
        scheduler: SchedulerMixin
    ) -> None:
        super().__init__()

        self.register_buffer("alphas_cumprod", scheduler.alphas_cumprod) # type: ignore
        self.register_buffer("final_alpha_cumprod", scheduler.final_alpha_cumprod) # type: ignore

        self.register_buffer("timestep", timestep)
        self.sample = nn.Parameter(sample.clone().detach())

        self.num_train_timesteps = scheduler.config.num_train_timesteps # type: ignore
        self.num_inference_steps = scheduler.num_inference_steps # type: ignore
        self.final_timestep = scheduler.config.steps_offset # type: ignore
        self.step_size = self.num_train_timesteps // self.num_inference_steps

    def __repr__(self) -> str:
        return ( "Sample(\n"
                f"  size={tuple(self.sample.size())},\n"
                f"  t={self.timestep.item()}\n"
                 ")")

    @property
    def prev_timestep(self) -> Tensor:
        return self.timestep - self.step_size

    @property
    def alpha_prod_t(self) -> Tensor:
        return self.alphas_cumprod[self.timestep]

    @property
    def alpha_prod_t_prev(self) -> Tensor:
        return self.alphas_cumprod[self.prev_timestep] if self.prev_timestep >= 0 else self.final_alpha_cumprod

    @property
    def beta_prod_t(self) -> Tensor:
        return 1 - self.alpha_prod_t

    @property
    def beta_prod_t_prev(self) -> Tensor:
        return 1 - self.alpha_prod_t_prev

    @property
    def variance(self) -> Tensor:
        return (self.beta_prod_t_prev / self.beta_prod_t) * (1 - self.alpha_prod_t / self.alpha_prod_t_prev)

    def std_dev_t(self, eta: float) -> Tensor:
        return eta * self.variance.sqrt()

    @property
    def timesteps(self) -> range:
        return range(self.timestep.int().item(), self.final_timestep - 1, -self.step_size) # type: ignore

    def unwrap(self) -> Tensor:
        return self.sample

    @torch.no_grad()
    def denoise(
        self,
        noise_pred_1: Tensor,
        noise_pred_2: Tensor | None = None,
        noise: Tensor | None = None,
        eta: float = 0.0,
    ) -> 'Sample':
        std = self.std_dev_t(eta)

        noise = std * torch.randn_like(self.sample) if noise is None else noise
        noise_pred_2 = noise_pred_1 if noise_pred_2 is None else noise_pred_2
        self.sample.copy_(
            self.alpha_prod_t_prev.sqrt() * self.tweedie(noise_pred_1)
            + (1 - self.alpha_prod_t_prev - std.pow(2)).sqrt() * noise_pred_2
            + noise
        )
        self.timestep = self.prev_timestep
        return self

    @torch.no_grad()
    def diffuse(
        self,
        noise: Tensor | None = None,
        target_timestep: int | None = None,
        step: int | None = None,
    ) -> 'Sample':
        assert target_timestep is None and step == 1, "Only step=1 is supported for now."

        self.timestep = self.timestep + self.step_size

        noise = torch.randn_like(self.sample) if noise is None else noise
        self.sample.copy_(
            (self.alpha_prod_t / self.alpha_prod_t_prev).sqrt() * self.sample
            + (1 - (self.alpha_prod_t / self.alpha_prod_t_prev)).sqrt() * noise
        )

        return self

    def tweedie(
        self,
        noise_pred: Tensor,
    ) -> Tensor:
        x = self.sample
        return (x - self.beta_prod_t.sqrt() * noise_pred) / self.alpha_prod_t.sqrt()

    def forward(self) -> Tensor:
        raise NotImplementedError
