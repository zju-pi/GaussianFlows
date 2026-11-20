import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

from torch.distributions import Normal
from tqdm import tqdm

from utils.file_utils import (
    get_output_dir,
    copy_source,
    setup_logging,
    setup_output_subdirs,
    set_seed,
)

from model.pvcnn_generation import PVCNN2Base
from datasets.shapenet_data_pc import ShapeNet15kPointClouds



class GaussianDiffusion:
    def __init__(self,betas, loss_type, model_mean_type, model_var_type):
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # initialize twice the actual length so we can keep running for eval
        # betas = np.concatenate([betas, np.full_like(betas[:int(0.2*len(betas))], betas[-1])])

        alphas = 1. - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(np.append(1., alphas_cumprod[:-1])).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1).float()

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = posterior_variance
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance)))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))



    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod.to(x_start.device), t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)
        assert noise.shape == x_start.shape
        return (
                self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise
        )


    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start +
                self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance.to(x_start.device), t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped.to(x_start.device), t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(self, denoise_fn, data, t, clip_denoised: bool, return_pred_xstart: bool):

        model_output = denoise_fn(data, t)


        if self.model_var_type in ['fixedsmall', 'fixedlarge']:
            # below: only log_variance is used in the KL computations
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
                'fixedlarge': (self.betas.to(data.device),
                               torch.log(torch.cat([self.posterior_variance[1:2], self.betas[1:]])).to(data.device)),
                'fixedsmall': (self.posterior_variance.to(data.device), self.posterior_log_variance_clipped.to(data.device)),
            }[self.model_var_type]
            model_variance = self._extract(model_variance, t, data.shape) * torch.ones_like(data)
            model_log_variance = self._extract(model_log_variance, t, data.shape) * torch.ones_like(data)
        else:
            raise NotImplementedError(self.model_var_type)

        if self.model_mean_type == 'eps':
            x_recon = self._predict_xstart_from_eps(data, t=t, eps=model_output)

            if clip_denoised:
                x_recon = torch.clamp(x_recon, -.5, .5)

            model_mean, _, _ = self.q_posterior_mean_variance(x_start=x_recon, x_t=data, t=t)
        else:
            raise NotImplementedError(self.loss_type)


        assert model_mean.shape == x_recon.shape == data.shape
        assert model_variance.shape == model_log_variance.shape == data.shape
        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon
        else:
            return model_mean, model_variance, model_log_variance

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps
        )

    ''' samples '''

    def p_sample(self, denoise_fn, data, t, noise_fn, clip_denoised=False, return_pred_xstart=False, use_var=True):
        """
        Sample from the model
        """
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(denoise_fn, data=data, t=t, clip_denoised=clip_denoised,
                                                                 return_pred_xstart=True)
        noise = noise_fn(size=data.shape, dtype=data.dtype, device=data.device)
        assert noise.shape == data.shape
        # no noise when t == 0
        nonzero_mask = torch.reshape(1 - (t == 0).float(), [data.shape[0]] + [1] * (len(data.shape) - 1))

        sample = model_mean
        if use_var:
            sample = sample + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        assert sample.shape == pred_xstart.shape
        return (sample, pred_xstart) if return_pred_xstart else sample


    def p_sample_loop(self, denoise_fn, shape, device,
                      noise_fn=torch.randn, constrain_fn=lambda x, t:x,
                      clip_denoised=True, max_timestep=None, keep_running=False):
        """
        Generate samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps

        """
        if max_timestep is None:
            final_time = self.num_timesteps
        else:
            final_time = max_timestep

        assert isinstance(shape, (tuple, list))
        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        for t in reversed(range(0, final_time if not keep_running else len(self.betas))):
            img_t = constrain_fn(img_t, t)
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t,t=t_, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised, return_pred_xstart=False).detach()


        assert img_t.shape == shape
        return img_t

    def get_ddim_schedule(self, num_steps):
        """Return DDIM state timesteps and the matching VE noise schedule.

        The returned tensors describe all states in sampling order. Therefore,
        they contain ``num_steps + 1`` entries: the initial noise state, one
        state after each DDIM update, and the final clean state (timestep -1).
        For a VP process, dividing a state by sqrt(alpha_bar) gives the VE
        parameterization with sigma = sqrt(1 - alpha_bar) / sqrt(alpha_bar).
        """
        if not 1 <= num_steps <= self.num_timesteps:
            raise ValueError(
                'num_steps must be in [1, {}], got {}'.format(
                    self.num_timesteps, num_steps
                )
            )

        # Include both ends of the training schedule and use exactly num_steps
        # network evaluations. Rounding avoids a bias toward early timesteps.
        if num_steps == 1:
            # A single denoising update must start at the terminal training
            # timestep, not timestep zero (np.linspace(..., 1) returns its start).
            ddim_timesteps = np.array([self.num_timesteps - 1], dtype=np.int64)
        else:
            ddim_timesteps = np.rint(
                np.linspace(0, self.num_timesteps - 1, num_steps)
            ).astype(np.int64)
        if len(np.unique(ddim_timesteps)) != num_steps:
            raise ValueError('DDIM timestep schedule contains duplicate entries')

        state_timesteps = np.concatenate([ddim_timesteps[::-1], [-1]])
        alpha_bar = torch.ones(num_steps + 1, dtype=torch.float32)
        non_clean = torch.from_numpy(state_timesteps[:-1].copy()).long()
        alpha_bar[:-1] = self.alphas_cumprod[non_clean]
        ve_schedule = torch.sqrt((1.0 - alpha_bar) / alpha_bar)
        ve_schedule[-1] = 0.0

        return (
            torch.from_numpy(ddim_timesteps).long(),
            torch.from_numpy(state_timesteps).long(),
            alpha_bar,
            ve_schedule,
        )

    def ddim_sample(self, denoise_fn, data, t, prev_t, noise_fn=torch.randn,
                    clip_denoised=False, eta=0.0):
        """Perform one DDIM update from timestep ``t`` to ``prev_t``."""
        eps = denoise_fn(data, t)
        alpha_t = self._extract(
            self.alphas_cumprod.to(data.device), t, data.shape
        )
        pred_xstart = (data - torch.sqrt(1.0 - alpha_t) * eps) / torch.sqrt(alpha_t)
        if clip_denoised:
            pred_xstart = torch.clamp(pred_xstart, -.5, .5)

        if prev_t < 0:
            alpha_prev = torch.ones_like(alpha_t)
        else:
            prev_t_batch = torch.full_like(t, prev_t)
            alpha_prev = self._extract(
                self.alphas_cumprod.to(data.device), prev_t_batch, data.shape
            )

        sigma = eta * torch.sqrt(
            torch.clamp(
                (1.0 - alpha_prev) / (1.0 - alpha_t)
                * (1.0 - alpha_t / alpha_prev),
                min=0.0,
            )
        )
        mean_pred = (
            torch.sqrt(alpha_prev) * pred_xstart
            + torch.sqrt(torch.clamp(1.0 - alpha_prev - sigma ** 2, min=0.0)) * eps
        )
        if eta > 0.0 and prev_t >= 0:
            mean_pred = mean_pred + sigma * noise_fn(
                size=data.shape, dtype=data.dtype, device=data.device
            )
        return mean_pred

    def ddim_sample_loop(self, denoise_fn, shape, device, num_steps=100,
                         eta=0.0, noise_fn=torch.randn,
                         constrain_fn=lambda x, t:x, clip_denoised=False,
                         return_trajectory=False):
        """Generate samples with DDIM, optionally retaining every VP state."""
        if eta < 0:
            raise ValueError('DDIM eta must be non-negative')
        assert isinstance(shape, (tuple, list))

        ddim_timesteps, state_timesteps, alpha_bar, ve_schedule = \
            self.get_ddim_schedule(num_steps)
        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        trajectory = [img_t.detach().cpu()] if return_trajectory else None

        for index in reversed(range(num_steps)):
            timestep = int(ddim_timesteps[index])
            prev_timestep = int(ddim_timesteps[index - 1]) if index > 0 else -1
            img_t = constrain_fn(img_t, timestep)
            t = torch.full(
                (shape[0],), timestep, dtype=torch.int64, device=device
            )
            img_t = self.ddim_sample(
                denoise_fn=denoise_fn,
                data=img_t,
                t=t,
                prev_t=prev_timestep,
                noise_fn=noise_fn,
                clip_denoised=clip_denoised,
                eta=eta,
            ).detach()
            if return_trajectory:
                trajectory.append(img_t.cpu())

        assert img_t.shape == shape
        if not return_trajectory:
            return img_t

        trajectory = torch.stack(trajectory, dim=1)
        assert trajectory.shape[:2] == (shape[0], num_steps + 1)
        trace = {
            'trajectory_vp': trajectory,
            'timesteps': state_timesteps,
            'alpha_bar': alpha_bar,
            've_schedule': ve_schedule,
        }
        return img_t, trace

    def get_edm_schedule(self, num_steps, device, schedule_type='karras',
                         sigma_min=None, sigma_max=None, rho=7.0):
        """Create an EDM-style decreasing VE sigma schedule plus sigma=0."""
        if num_steps < 2:
            raise ValueError('EDM sampling requires at least two steps')
        if rho <= 0:
            raise ValueError('EDM rho must be positive')

        training_sigmas = (
            self.sqrt_one_minus_alphas_cumprod
            / self.sqrt_alphas_cumprod
        )
        train_sigma_min = float(training_sigmas[0])
        train_sigma_max = float(training_sigmas[-1])
        sigma_min = train_sigma_min if sigma_min is None else sigma_min
        sigma_max = train_sigma_max if sigma_max is None else sigma_max
        if not 0 < sigma_min < sigma_max:
            raise ValueError(
                'Expected 0 < sigma_min < sigma_max, got {} and {}'.format(
                    sigma_min, sigma_max
                )
            )
        if sigma_min < train_sigma_min or sigma_max > train_sigma_max:
            raise ValueError(
                'EDM sigma range [{:.6g}, {:.6g}] lies outside the model '
                'training range [{:.6g}, {:.6g}]'.format(
                    sigma_min, sigma_max, train_sigma_min, train_sigma_max
                )
            )

        ramp = torch.linspace(0, 1, num_steps, device=device)
        if schedule_type == 'karras':
            min_inv_rho = sigma_min ** (1.0 / rho)
            max_inv_rho = sigma_max ** (1.0 / rho)
            sigmas = (
                max_inv_rho + ramp * (min_inv_rho - max_inv_rho)
            ) ** rho
        elif schedule_type == 'exponential':
            sigmas = torch.exp(
                torch.linspace(
                    np.log(sigma_max), np.log(sigma_min), num_steps,
                    device=device,
                )
            )
        elif schedule_type == 'vp':
            indices = torch.linspace(
                self.num_timesteps - 1, 0, num_steps, device=device
            ).round().long()
            sigmas = training_sigmas.to(device)[indices]
        else:
            raise ValueError('Unknown EDM schedule: {}'.format(schedule_type))

        return torch.cat([sigmas.float(), sigmas.new_zeros(1).float()])

    def _edm_model_timestep(self, sigma, device):
        """Map a continuous VE sigma to the nearest trained VP timestep."""
        training_sigmas = (
            self.sqrt_one_minus_alphas_cumprod
            / self.sqrt_alphas_cumprod
        ).to(device)
        return int(torch.argmin(torch.abs(training_sigmas - sigma)).item())

    def _edm_derivative(self, denoise_fn, data_ve, sigma,
                        clip_denoised=False):
        """Evaluate dx/dsigma in VE space using the VP epsilon network."""
        alpha_bar = 1.0 / (1.0 + sigma ** 2)
        timestep = self._edm_model_timestep(sigma, data_ve.device)
        t = torch.full(
            (data_ve.shape[0],), timestep, dtype=torch.int64,
            device=data_ve.device,
        )
        # x_VP = sqrt(alpha_bar) * x_VE. The trained network predicts epsilon,
        # which equals the VE probability-flow ODE derivative dx/dsigma.
        data_vp = data_ve * torch.sqrt(alpha_bar)
        eps = denoise_fn(data_vp, t)
        denoised = data_ve - sigma * eps
        if clip_denoised:
            denoised = torch.clamp(denoised, -.5, .5)
        derivative = (data_ve - denoised) / sigma
        return derivative, denoised, timestep

    def edm_sample_loop(self, denoise_fn, shape, device, num_steps=100,
                        solver='heun', schedule_type='karras', rho=7.0,
                        sigma_min=None, sigma_max=None, noise_fn=torch.randn,
                        clip_denoised=False, return_trajectory=False):
        """EDM-style probability-flow ODE sampling directly in VE space."""
        if solver not in ('euler', 'heun'):
            raise ValueError('EDM solver must be euler or heun')
        assert isinstance(shape, (tuple, list))
        sigmas = self.get_edm_schedule(
            num_steps=num_steps,
            device=device,
            schedule_type=schedule_type,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
        )

        data_ve = noise_fn(
            size=shape, dtype=torch.float, device=device
        ) * sigmas[0]
        trajectory = [data_ve.detach().cpu()] if return_trajectory else None
        model_timesteps = []

        for index in range(num_steps):
            sigma_cur = sigmas[index]
            sigma_next = sigmas[index + 1]
            derivative, _, timestep = self._edm_derivative(
                denoise_fn, data_ve, sigma_cur, clip_denoised
            )
            model_timesteps.append(timestep)
            step = sigma_next - sigma_cur
            proposal = data_ve + step * derivative

            # EDM's second-order correction is skipped for the terminal step,
            # because the epsilon model was never trained at sigma=0.
            if solver == 'heun' and float(sigma_next) > 0:
                next_derivative, _, _ = self._edm_derivative(
                    denoise_fn, proposal, sigma_next, clip_denoised
                )
                data_ve = data_ve + step * (
                    0.5 * derivative + 0.5 * next_derivative
                )
            else:
                data_ve = proposal
            data_ve = data_ve.detach()
            if return_trajectory:
                trajectory.append(data_ve.cpu())

        state_timesteps = torch.tensor(
            model_timesteps + [-1], dtype=torch.long
        )
        alpha_bar = 1.0 / (1.0 + sigmas.detach().cpu() ** 2)
        metadata = {
            'timesteps': state_timesteps,
            'alpha_bar': alpha_bar,
            've_schedule': sigmas.detach().cpu(),
            'solver': solver,
            'schedule_type': schedule_type,
            'rho': rho,
        }
        if not return_trajectory:
            return data_ve

        trajectory_ve = torch.stack(trajectory, dim=1)
        assert trajectory_ve.shape[:2] == (shape[0], num_steps + 1)
        metadata['trajectory_ve'] = trajectory_ve
        # Also retain the equivalent VP states for compatibility with earlier
        # trajectory consumers. Both trajectories end at the same clean x0.
        scale = torch.sqrt(alpha_bar).reshape(1, -1, *([1] * (len(shape) - 1)))
        metadata['trajectory_vp'] = trajectory_ve * scale
        return data_ve, metadata

    def reconstruct(self, x0, t, denoise_fn, noise_fn=torch.randn, constrain_fn=lambda x, t:x):

        assert t >= 1

        t_vec = torch.empty(x0.shape[0], dtype=torch.int64, device=x0.device).fill_(t-1)
        encoding = self.q_sample(x0, t_vec)

        img_t = encoding

        for k in reversed(range(0,t)):
            img_t = constrain_fn(img_t, k)
            t_ = torch.empty(x0.shape[0], dtype=torch.int64, device=x0.device).fill_(k)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, noise_fn=noise_fn,
                                  clip_denoised=False, return_pred_xstart=False, use_var=True).detach()


        return img_t


class PVCNN2(PVCNN2Base):
    sa_blocks = [
        ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
        ((64, 3, 16), (256, 0.2, 32, (64, 128))),
        ((128, 3, 8), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (256, 256, 512))),
    ]
    fp_blocks = [
        ((256, 256), (256, 3, 8)),
        ((256, 256), (256, 3, 8)),
        ((256, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]

    def __init__(self, num_classes, embed_dim, use_att,dropout, extra_feature_channels=3, width_multiplier=1,
                 voxel_resolution_multiplier=1):
        super().__init__(
            num_classes=num_classes, embed_dim=embed_dim, use_att=use_att,
            dropout=dropout, extra_feature_channels=extra_feature_channels,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )



class Model(nn.Module):
    def __init__(self, args, betas, loss_type: str, model_mean_type: str, model_var_type:str):
        super(Model, self).__init__()
        self.diffusion = GaussianDiffusion(betas, loss_type, model_mean_type, model_var_type)

        self.model = PVCNN2(num_classes=args.nc, embed_dim=args.embed_dim, use_att=args.attention,
                            dropout=args.dropout, extra_feature_channels=0)

    def prior_kl(self, x0):
        return self.diffusion._prior_bpd(x0)

    def all_kl(self, x0, clip_denoised=True):
        total_bpd_b, vals_bt, prior_bpd_b, mse_bt =  self.diffusion.calc_bpd_loop(self._denoise, x0, clip_denoised)

        return {
            'total_bpd_b': total_bpd_b,
            'terms_bpd': vals_bt,
            'prior_bpd_b': prior_bpd_b,
            'mse_bt':mse_bt
        }


    def _denoise(self, data, t):
        B, D,N= data.shape
        assert data.dtype == torch.float
        assert t.shape == torch.Size([B]) and t.dtype == torch.int64

        out = self.model(data, t)

        assert out.shape == torch.Size([B, D, N])
        return out

    def get_loss_iter(self, data, noises=None):
        B, D, N = data.shape
        t = torch.randint(0, self.diffusion.num_timesteps, size=(B,), device=data.device)

        if noises is not None:
            noises[t!=0] = torch.randn((t!=0).sum(), *noises.shape[1:]).to(noises)

        losses = self.diffusion.p_losses(
            denoise_fn=self._denoise, data_start=data, t=t, noise=noises)
        assert losses.shape == t.shape == torch.Size([B])
        return losses

    def gen_samples(self, shape, device, sampler='edm', noise_fn=torch.randn,
                    constrain_fn=lambda x, t:x, clip_denoised=False,
                    num_steps=100, eta=0.0, return_trajectory=False,
                    edm_solver='heun', edm_schedule='karras', edm_rho=7.0,
                    sigma_min=None, sigma_max=None):
        if sampler == 'ddim':
            return self.diffusion.ddim_sample_loop(
                self._denoise,
                shape=shape,
                device=device,
                num_steps=num_steps,
                eta=eta,
                noise_fn=noise_fn,
                constrain_fn=constrain_fn,
                clip_denoised=clip_denoised,
                return_trajectory=return_trajectory,
            )
        if sampler == 'edm':
            return self.diffusion.edm_sample_loop(
                self._denoise,
                shape=shape,
                device=device,
                num_steps=num_steps,
                solver=edm_solver,
                schedule_type=edm_schedule,
                rho=edm_rho,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                noise_fn=noise_fn,
                clip_denoised=clip_denoised,
                return_trajectory=return_trajectory,
            )
        raise ValueError('Unknown sampler: {}'.format(sampler))

    def reconstruct(self, x0, t, constrain_fn=lambda x, t:x):

        return self.diffusion.reconstruct(x0, t, self._denoise, constrain_fn=constrain_fn)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def multi_gpu_wrapper(self, f):
        self.model = f(self.model)


def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == 'linear':
        betas = np.linspace(b_start, b_end, time_num)
    elif schedule_type == 'warm0.1':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.2':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.5':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    else:
        raise NotImplementedError(schedule_type)
    return betas



#############################################################################

def get_dataset(dataroot, npoints,category,use_mask=False):
    tr_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
        categories=[category], split='train',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        random_subsample=True, use_mask = use_mask)
    te_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
        categories=[category], split='val',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        all_points_mean=tr_dataset.all_points_mean,
        all_points_std=tr_dataset.all_points_std,
        use_mask=use_mask
    )
    return tr_dataset, te_dataset



def generate(model, opt):

    _, test_dataset = get_dataset(opt.dataroot, opt.npoints, opt.category)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size,
                                                  shuffle=True, num_workers=int(opt.workers), drop_last=False)

    with torch.no_grad():

        samples = []
        ref = []
        trajectory_chunks = {}
        normalization_means = []
        normalization_stds = []
        sampling_metadata = None

        for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Generating Samples'):

            x = data['test_points'].transpose(1,2)
            m, s = data['mean'].float(), data['std'].float()

            generated = model.gen_samples(
                x.shape,
                'cuda',
                sampler=opt.sampler,
                clip_denoised=False,
                num_steps=opt.sampling_steps,
                eta=opt.ddim_eta,
                return_trajectory=opt.save_trajectory,
                edm_solver=opt.edm_solver,
                edm_schedule=opt.edm_schedule,
                edm_rho=opt.edm_rho,
                sigma_min=opt.sigma_min,
                sigma_max=opt.sigma_max,
            )
            if opt.save_trajectory:
                gen, trace = generated
                # Model tensors are [B, 3, N]; store point clouds as [B, N, 3].
                for key in ('trajectory_vp', 'trajectory_ve'):
                    if key in trace:
                        trajectory_chunks.setdefault(key, []).append(
                            trace[key].transpose(-1, -2).contiguous()
                        )
                normalization_means.append(m.cpu())
                normalization_stds.append(s.cpu())
                sampling_metadata = {
                    key: value.cpu() if torch.is_tensor(value) else value
                    for key, value in trace.items()
                    if not key.startswith('trajectory_')
                }
            else:
                gen = generated
            gen = gen.detach().cpu()

            gen = gen.transpose(1,2).contiguous()
            x = x.transpose(1,2).contiguous()


            gen = gen * s + m
            x = x * s + m
            samples.append(gen)
            ref.append(x)

            break

        samples = torch.cat(samples, dim=0)
        ref = torch.cat(ref, dim=0)

        torch.save(samples, opt.eval_path)

        # The schedule is tiny and useful even when the full trajectory is not
        # requested. Its entries align with initial noise -> final sample.
        if sampling_metadata is None:
            if opt.sampler == 'ddim':
                _, timesteps, alpha_bar, ve_schedule = \
                    model.diffusion.get_ddim_schedule(opt.sampling_steps)
                sampling_metadata = {
                    'timesteps': timesteps,
                    'alpha_bar': alpha_bar,
                    've_schedule': ve_schedule,
                }
            else:
                ve_schedule = model.diffusion.get_edm_schedule(
                    opt.sampling_steps,
                    device='cpu',
                    schedule_type=opt.edm_schedule,
                    sigma_min=opt.sigma_min,
                    sigma_max=opt.sigma_max,
                    rho=opt.edm_rho,
                )
                timesteps = [
                    model.diffusion._edm_model_timestep(sigma, 'cpu')
                    for sigma in ve_schedule[:-1]
                ] + [-1]
                sampling_metadata = {
                    'timesteps': torch.tensor(timesteps, dtype=torch.long),
                    'alpha_bar': 1.0 / (1.0 + ve_schedule ** 2),
                    've_schedule': ve_schedule,
                    'solver': opt.edm_solver,
                    'schedule_type': opt.edm_schedule,
                    'rho': opt.edm_rho,
                }
        schedule_path = os.path.join(
            str(Path(opt.eval_path).parent),
            '{}_schedule.pth'.format(opt.sampler),
        )
        schedule_payload = {
            **sampling_metadata,
            'sampler': opt.sampler,
            'num_train_timesteps': model.diffusion.num_timesteps,
            'num_sampling_steps': opt.sampling_steps,
            've_formula': 'sqrt(1 - alpha_bar) / sqrt(alpha_bar)',
        }
        if opt.sampler == 'ddim':
            schedule_payload['ddim_eta'] = opt.ddim_eta
        torch.save(schedule_payload, schedule_path)

        if opt.save_trajectory:
            trajectory_path = os.path.join(
                str(Path(opt.eval_path).parent), 'sampling_trajectory.pth'
            )
            saved_trajectories = {
                key: torch.cat(chunks, dim=0)
                for key, chunks in trajectory_chunks.items()
            }
            torch.save(
                {
                    **saved_trajectories,
                    **sampling_metadata,
                    'normalization_mean': torch.cat(normalization_means, dim=0),
                    'normalization_std': torch.cat(normalization_stds, dim=0),
                    'sampler': opt.sampler,
                    'num_train_timesteps': model.diffusion.num_timesteps,
                    'num_sampling_steps': opt.sampling_steps,
                    'layout': 'batch, state, point, xyz',
                    'space': 'normalized VP and VE' if opt.sampler == 'edm'
                             else 'normalized VP',
                    've_conversion': 'trajectory_vp / sqrt(alpha_bar)',
                },
                trajectory_path,
            )



    return ref


def main(opt):

    if opt.category == 'airplane':
        opt.beta_start = 1e-5
        opt.beta_end = 0.008
        opt.schedule_type = 'warm0.1'

    exp_id = os.path.splitext(os.path.basename(__file__))[0]
    dir_id = os.path.dirname(__file__)
    output_dir = get_output_dir(dir_id, exp_id)
    copy_source(__file__, output_dir)
    logger = setup_logging(output_dir)

    outf_syn, = setup_output_subdirs(output_dir, 'syn')

    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    model = Model(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type)

    if opt.cuda:
        model.cuda()

    model = model.cuda()

    model.eval()

    with torch.no_grad():

        logger.info("Resume Path:%s" % opt.model)

        from collections import OrderedDict
        resumed_param = torch.load(opt.model)
        state_dict = resumed_param["model_state"]
        new_state_dict = OrderedDict(
            (k.replace("module.", ""), v)
            for k, v in state_dict.items()
        )
        model.load_state_dict(new_state_dict)


        ref = None
        if opt.generate:
            opt.eval_path = os.path.join(outf_syn, 'samples.pth')
            Path(opt.eval_path).parent.mkdir(parents=True, exist_ok=True)
            ref=generate(model, opt)
            

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='assets/datasets/pvd/ShapeNetCore.v2.PC15k')
    parser.add_argument('--category', default='chair')

    parser.add_argument('--batch_size', type=int, default=50, help='input batch size')
    parser.add_argument('--workers', type=int, default=16, help='workers')
    parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')

    parser.add_argument('--generate',default=True)
    parser.add_argument('--eval_gen', default=True)

    parser.add_argument('--nc', default=3)
    parser.add_argument('--npoints', default=2048)
    '''model'''
    parser.add_argument('--beta_start', type=float, default=0.0001)
    parser.add_argument('--beta_end', type=float, default=0.02)
    parser.add_argument('--schedule_type', default='linear')
    parser.add_argument('--time_num', type=int, default=1000)
    parser.add_argument(
        '--sampler', choices=('edm', 'ddim'), default='edm',
        help='sampling method (default: EDM-style VE ODE)'
    )
    parser.add_argument(
        '--sampling_steps', '--sampling-steps', '--ddim_steps', '--ddim-steps',
        dest='sampling_steps', type=int, default=100,
        help='number of ODE/DDIM updates (default: 100)'
    )
    parser.add_argument(
        '--ddim_eta', '--ddim-eta', dest='ddim_eta', type=float,
        default=0.0, help='DDIM stochasticity; 0 gives deterministic DDIM'
    )
    parser.add_argument(
        '--save_trajectory', '--save-trajectory', dest='save_trajectory',
        action='store_true',
        help='save all sampling states, including the initial noise state'
    )
    parser.add_argument(
        '--edm_solver', '--edm-solver', dest='edm_solver',
        choices=('euler', 'heun'), default='heun',
        help='VE ODE solver; Heun is the EDM second-order method'
    )
    parser.add_argument(
        '--edm_schedule', '--edm-schedule', dest='edm_schedule',
        choices=('karras', 'exponential', 'vp'), default='karras',
        help='VE sigma discretization (default: Karras)'
    )
    parser.add_argument(
        '--edm_rho', '--edm-rho', dest='edm_rho', type=float, default=7.0,
        help='Karras schedule curvature (default: 7)'
    )
    parser.add_argument(
        '--sigma_min', '--sigma-min', dest='sigma_min', type=float, default=None,
        help='VE sigma minimum; defaults to the lowest training sigma'
    )
    parser.add_argument(
        '--sigma_max', '--sigma-max', dest='sigma_max', type=float, default=None,
        help='VE sigma maximum; defaults to the highest training sigma'
    )

    #params
    parser.add_argument('--attention', default=True)
    parser.add_argument('--dropout', default=0.1)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--loss_type', default='mse')
    parser.add_argument('--model_mean_type', default='eps')
    parser.add_argument('--model_var_type', default='fixedsmall')


    parser.add_argument('--model', default='',required=True, help="path to model (to continue training)")

    '''eval'''

    parser.add_argument('--eval_path',
                        default='')

    # parser.add_argument('--manualSeed', default=42, type=int, help='random seed')
    parser.add_argument('--manualSeed', type=int, help='random seed')

    parser.add_argument('--gpu', type=int, default=0, metavar='S', help='gpu id (default: 0)')

    opt = parser.parse_args()

    if torch.cuda.is_available():
        opt.cuda = True
    else:
        opt.cuda = False

    return opt
if __name__ == '__main__':
    opt = parse_args()
    set_seed(opt)

    main(opt)
