from IPython import embed
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from functools import partial
import numpy as np
from inspect import isfunction
from tqdm import tqdm
import matplotlib.pyplot as plt
import einops
import random
from datetime import datetime
import os

from data.dataset import DESS2TSEDataset
from models.network import MyNetwork
from models.guided_diffusion_modules.unet import UNet
from models.model import Palette, MyPalette
from ptoa.data.knee_monai import SliceDataset, KneeDataset

CONFIG = {
    'img_size': 256,
    # 'task': 'bone_premask',
    'task': 'inpaint_bone', # 'no_premask', 'inpaint_roi'  'bone_premask'
    'schedule': 'linear', # 'linear', 'warmup10', 'warmup50', 'const', 'jsd' 'quad'
    'schedule_start': 1e-6,
    'schedule_end': 1e-2,
    'schedule_consine_s': 8e-3,
    'n_timesteps': 500,
    'dropout': 0.2,
    'loss_fn': 'mse',
    'lr': 2.5e-5,
    'batch_size': 8,
    'n_epochs': 1000,
}

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape=(1,1,1,1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# beta_schedule function
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas

class Colorize(nn.Module):
    def __init__(self, device, config=CONFIG):
        super(Colorize, self).__init__()
        self.device = device
        self.config = config
        self.net = UNet(
            image_size=self.config['img_size'],
            in_channel=2,
            inner_channel=64,
            out_channel=1,
            res_blocks=2,
            attn_res=[16],
            num_head_channels=32,
            dropout=self.config['dropout'],
        ).to(device)
        if self.config['loss_fn'] == 'mse':
            self.loss_fn = F.mse_loss
        else:
            raise NotImplementedError
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.config['lr'])
        self.set_noise_schedule()

    def set_noise_schedule(self):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)
        betas = make_beta_schedule(
            schedule=self.config['schedule'],
            n_timestep=self.config['n_timesteps'],
            linear_start=self.config['schedule_start'],
            linear_end=self.config['schedule_end'],
        )
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas

        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1., gammas[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

    def predict_start_from_noise(self, y_t, t, noise):
        return (
            extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -
            extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def q_posterior(self, y_0_hat, y_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
            extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool):
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        y_0_hat = self.predict_start_from_noise(
                y_t, t=t, noise=self.net(torch.cat([self.x, y_t], dim=1), noise_level))

        if clip_denoised:
            y_0_hat.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t)
        return model_mean, posterior_log_variance

    def q_sample(self, sample_gammas, noise=None):
        noise = default(noise, lambda: torch.randn_like(self.y))
        return (
            sample_gammas.sqrt() * self.y +
            (1 - sample_gammas).sqrt() * noise
        )

    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True):
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised)
        noise = torch.randn_like(y_t) if any(t>0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def restoration(self, y_t=None, sample_num=8):
        b = self.config['batch_size']
        assert self.config['n_timesteps'] > sample_num, 'num_timesteps must greater than sample_num'
        sample_inter = (self.config['n_timesteps']//sample_num)
        
        y_t = default(y_t, lambda: torch.randn_like(self.x))
        ret_arr = y_t
        for i in tqdm(reversed(range(0, self.config['n_timesteps'])), desc='sampling loop time step', total=self.config['n_timesteps']):
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            y_t = self.p_sample(y_t, t)
            if 'inpaint' in self.config['task']:
                y_t = self.y*(1.-self.mask) + self.mask*y_t
            if i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t], dim=0)
        return y_t, ret_arr

    def forward(self, noise=None):
        # sampling from p(gammas)
        b = self.config['batch_size']
        t = torch.randint(1, self.config['n_timesteps'], (b,), device=self.y.device).long()
        gamma_t1 = extract(self.gammas, t-1, x_shape=(1, 1))
        sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=self.y.device) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(self.y))
        y_noisy = self.q_sample(
            sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)

        if 'inpaint' in self.config['task']:
            noise_hat = self.net(torch.cat([self.x, y_noisy*self.mask+(1.-self.mask)*self.y], dim=1), sample_gammas)
            loss = self.loss_fn(self.mask*noise, self.mask*noise_hat)
        else:
            noise_hat = self.net(torch.cat([self.x, y_noisy], dim=1), sample_gammas)
            loss = self.loss_fn(noise, noise_hat)
        return loss

    def set_input(self, batch):
        self.x = batch['tse'].to(self.device)
        self.y = batch['tse'].to(self.device)
        # if self.config['task'] == 'inpaint_bone', 'inpaint_roi':...
        self.mask = (batch['bone'] > 0).to(torch.float32).to(self.device)
        self.mask_nodil = (batch['bone_nodil'] > 0).to(torch.float32).to(self.device)
        self.id = batch['id']

    def train_step(self):
        self.net.train()
        self.optim.zero_grad()
        loss = self.forward()
        loss.backward()
        self.optim.step()
        # scheduler.step???
        return loss.detach().cpu()

    def val_step(self):
        self.net.eval()
        with torch.no_grad():
            self.y_pred, self.y_t = self.restoration()

    def test(self):
        pass
    
if __name__ == '__main__':
    savepath = f"./runs_{datetime.strftime(datetime.now(), '%y%m%d_%H%M')}_{CONFIG['task']}/"
    os.makedirs(savepath, mode=0o755, exist_ok=True)

    ds = DESS2TSEDataset(config=CONFIG)
    dl = DataLoader(ds, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    c = Colorize(device='cuda', config=CONFIG)

    n_epochs = CONFIG['n_epochs']
    for epoch_ndx in range(1, n_epochs+1):
        losses = []
        for batch in dl:
            c.set_input(batch)
            loss = c.train_step()
            losses.append(loss)
        print(f"e{epoch_ndx}: {torch.tensor(losses).mean()}")
        if epoch_ndx == 1 or epoch_ndx % 5 == 0:
            for batch in dl:
                c.set_input(batch)
                c.val_step()
                out = torch.stack((c.x, c.y_pred, c.y), dim=0)
                fig, ax = plt.subplots(1, 1, figsize=(50, 15))
                mat = ax.imshow(einops.rearrange(out.cpu(), 'out b () h w -> (out h) (b w)'), cmap='gray')
                for i in range(8):
                    ax.text(256 * i, 20, f"MSE={F.mse_loss(c.y_pred[i], c.y[i]).item():.4f}", c='w')
                plt.colorbar(mat, ax=ax)
                plt.savefig(f'{savepath}E{epoch_ndx:04d}.png')
                plt.close()
                print(f"E{epoch_ndx}: MSE={F.mse_loss(c.y_pred, c.y).item():.4f}")
                break
        if epoch_ndx == 1 or epoch_ndx % 20 == 0:
            save_filename = f'{savepath}E{epoch_ndx:04d}.pth'
            state_dict = c.net.state_dict()
            for key, param in state_dict.items():
                state_dict[key] = param.cpu()
            torch.save(state_dict, save_filename)