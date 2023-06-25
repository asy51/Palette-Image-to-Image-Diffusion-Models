import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from functools import partial
import numpy as np
from inspect import isfunction
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict
import einops

from models.network import MyNetwork
from models.guided_diffusion_modules.unet import UNet
from models.model import Palette, MyPalette
from ptoa.data.knee_monai import SliceDataset, KneeDataset, Knee
from myrun import DESS2TSEDataset, Colorize, make_beta_schedule
import myrun

BATCH_SIZE=1

ds = DESS2TSEDataset(bmel=True, count=20)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

c = Colorize(device='cuda')
model_path = 'runs_1000/E0080.pth'
state_dict = torch.load(model_path)
c.net.load_state_dict(state_dict)

for batch in dl:
    c.set_input(batch)
    batch_size = c.x.shape[0]
    # restoration
    # ytees = []
    y_t = torch.randn_like(c.x, device=c.device)
    for i in tqdm(reversed(range(0, c.num_timesteps)), desc='sampling loop time step', total=c.num_timesteps):
        t = torch.full((batch_size,), i, device=c.device, dtype=torch.long)
        y_t = c.p_sample(y_t, t, y_cond=c.x)
        # ytees.append(y_t.clone())
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    mat = ax.imshow(y_t.cpu()[0,0], cmap='gray')
    loss = F.mse_loss(y_t, c.y).item()
    ax.text(0, 20, f"MSE={loss:.4f}", c='w')
    plt.colorbar(mat, ax=ax)
    plt.savefig(f"{c.id}.png")
    print(f"{c.id} {loss:.4f}")
    