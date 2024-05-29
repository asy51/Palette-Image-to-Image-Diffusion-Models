from IPython import embed
from argparse import Namespace
from data.dataset import FastInpaintBMELDataset, MoonCometInpaintDataset, MoonCometBoneInpaintDataset, MoonCometTranslateDataset
from torch.utils.data import DataLoader
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import tqdm
import os

import core.praser as Praser
from models.network import Network

import argparse

# if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('-x', '--experiment', type=str)
parser.add_argument('-d', '--device', type=int)
parser.add_argument('-e', '--epoch', type=int)
parser.add_argument('--ema', type=str, default='')
args = parser.parse_args()

device = f'cuda:{args.device}'
# EMA = ''
root = Path('experiments/')
root /= args.experiment
save = root / f'preds_{args.epoch}{args.ema}/'

if os.path.exists(save):
    print(f"{save} already exists")
else:
    os.makedirs(save,mode=0o755, exist_ok=True)
    print(f"{save} created")

palette_args = Namespace(config=root / 'config.json', phase='test', gpu_ids=None, batch=16, debug=False)
opt = Praser.parse_test(palette_args)

model_args = opt["model"]["which_networks"][0]["args"]
model = Network(**model_args)
model_pth = Path(root) / 'checkpoint' / f'{args.epoch}_Network{args.ema}.pth'
state_dict = torch.load(model_pth)
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.set_new_noise_schedule(device=device, phase='test')

ds = MoonCometTranslateDataset(slc_has_bmel=False)

div = len(ds) // 3
ndx_start = div * args.device
ndx_end = div * (args.device+1)
if args.device == 2:
    ndx_end = None
ds.slices = ds.slices[ndx_start:ndx_end]
print(len(ds), ndx_start, ndx_end)

dl = DataLoader(ds, batch_size=palette_args.batch)
ret = {}
for batch in tqdm.tqdm(dl):
    if all(os.path.exists(save / path.replace('png','pt')) for path in batch['path']):
        print(f'all {batch["path"]} exists')
        continue
    batch_size = len(batch['path'])
    model.output, model.visuals = model.restoration(
        batch['cond_image'].to(device),
        y_t=batch['cond_image'].to(device),
        y_0=batch['gt_image'].to(device),
        # mask=batch['mask'].to(device),
        sample_num=8
    )

    fig, ax = plt.subplots(4, batch_size, figsize=(10 * batch_size / 5, 10))
    for ndx in range(batch_size):
        sample = {
            'gt': batch['gt_image'][ndx,0],
            'pred': model.output[ndx,0].cpu(),
            # 'bmel': batch['bmel'][ndx,0].cpu(),
            # 'mask': batch['mask'][ndx,0]
        }
        ret[batch['path'][ndx]] = sample
        savepath = save / batch['path'][ndx].replace('png','pt')
        torch.save(sample['pred'], savepath)
        print(f"saved to {savepath}")
    #     im = ax[0,ndx].imshow(batch['cond_image'][ndx,0], 'gray')
    #     plt.colorbar(im, ax=ax[0,ndx])
    #     im = ax[1,ndx].imshow(batch['gt_image'][ndx,0], 'gray')
    #     plt.colorbar(im, ax=ax[1,ndx])
    #     im = ax[2,ndx].imshow(model.output[ndx,0].cpu(), 'gray')
    #     plt.colorbar(im, ax=ax[2,ndx])
    #     im = ax[3,ndx].imshow(model.output[ndx,0].cpu() - batch['gt_image'][ndx,0], 'gray')
    #     plt.colorbar(im, ax=ax[3,ndx])
    # plt.show()