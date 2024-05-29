import glob
import torch
import matplotlib.pyplot as plt
from ptoa.data.knee_monai import SliceDataset, KneeDataset, Knee

from IPython import embed
from argparse import Namespace
from data.dataset import MoonCometDataset
from torch.utils.data import DataLoader
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import tqdm
import os
import pandas as pd
import core.praser as Praser
from models.network import Network

import argparse
import gc

parser = argparse.ArgumentParser()
parser.add_argument('-x', '--experiment', type=str)
parser.add_argument('-d', '--device', type=int)
parser.add_argument('-e', '--epoch', type=int, )
parser.add_argument('-b', '--batch_size', type=int, default=1)
parser.add_argument('--ema', action='store_true', default='')
pred_args = parser.parse_args()
    
args = argparse.Namespace(
    experiment=pred_args.experiment,
    device=0,
    epoch=pred_args.epoch,
    ema='ema' if pred_args.ema else ''
)
device=f'cuda:{pred_args.device}'

root = Path('experiments/')
root /= args.experiment
save = root / f'preds_{args.epoch}{args.ema}/'

palette_args = Namespace(config=root / 'config.json', phase='test', gpu_ids=None, batch=16, debug=False)
opt = Praser.parse_test(palette_args)
img_size = opt['datasets']['train']['which_dataset']['args']['img_size']
task=opt['datasets']['train']['which_dataset']['args']['task']

model_args = opt["model"]["which_networks"][0]["args"]
model = Network(**model_args)
model_pth = Path(root) / 'checkpoint' / f'{args.epoch}_Network{args.ema}.pth'
state_dict = torch.load(model_pth)
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.set_new_noise_schedule(device=device, phase='test')
# load data
df = pd.read_csv('/home/yua4/bip_submission/hakan_bmel_intra_nifti.csv', na_values='None')
df = df[df['base'] != 'comet-patient-ccf-015-20210920-knee']
knees_test = [Knee(base) for base in df['base']]
ds = MoonCometDataset(img_size=img_size, knees=knees_test, task=task, k_mask=['BONE_TSE', 'BMELT'])
ds.slices = [d for d in ds.slices if d['BMELT'].sum() > 0]
print(f'n slices: {len(ds)}')
dl = DataLoader(ds, batch_size=pred_args.batch_size)

if os.path.exists(save):
    print(f"{save} already exists")
else:
    os.makedirs(save,mode=0o755, exist_ok=True)
    print(f"{save} created")

for batch in tqdm.tqdm(dl):
    model.output, model.visuals = model.restoration(
        batch['cond_image'].to(device),
        y_t=batch['cond_image'].to(device),
        y_0=batch['gt_image'].to(device),
        sample_num=2
    )
    current_batch_size = batch['gt_image'].shape[0]
    for ndx in range(pred_args.batch_size):
        sample = {
            'gt': batch['gt_image'][ndx,0],
            'pred': model.output[ndx,0].cpu(),
            'path': batch['path'][ndx],
            # 'bmel': batch['bmel'][ndx,0].cpu(),
            # 'mask': batch['mask'][ndx,0]
        }
        # VIZ
        # fig, ax = plt.subplots(4, pred_args.batch_size, figsize=(10 * pred_args.batch_size / 5, 10), squeeze=False)
        # for sample_ndx in range(pred_args.batch_size):
        #     im = ax[0,sample_ndx].imshow(batch['cond_image'][sample_ndx,0], 'gray')
        #     plt.colorbar(im, ax=ax[0,sample_ndx])
        #     im = ax[1,sample_ndx].imshow(batch['gt_image'][sample_ndx,0], 'gray')
        #     plt.colorbar(im, ax=ax[1,sample_ndx])
        #     im = ax[2,sample_ndx].imshow(model.output[sample_ndx,0].cpu(), 'gray')
        #     plt.colorbar(im, ax=ax[2,sample_ndx])
        #     im = ax[3,sample_ndx].imshow(model.output[sample_ndx,0].cpu() - batch['gt_image'][sample_ndx,0], 'gray')
        #     plt.colorbar(im, ax=ax[3,sample_ndx])
        # plt.show()
        # SAVE
        savepath = save / sample['path'].replace('png','pt')
        torch.save(sample['pred'], savepath)
        print(f"saved to {savepath}")

    # CLEAN
    del model.output; model.output = None
    del model.visuals; model.visuals = None
    gc.collect()
    torch.cuda.empty_cache()
