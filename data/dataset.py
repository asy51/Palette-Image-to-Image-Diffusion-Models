from IPython import embed

import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)
# from .util.mask import brush_stroke_mask, get_irregular_mask

import monai.transforms as MT
import einops
import h5py
import pandas as pd
import glob
from tqdm import tqdm
tqdm.pandas()

from ptoa.data.knee_monai import SliceDataset, KneeDataset

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)


class DESS2TSEDataset(SliceDataset):
    def __init__(self, kds=None, img_size=256, **kwargs):
        if kds is None:
            kds = KneeDataset()
            kds.knees = [knee for knee in kds.knees if all(knee.path[k] for k in ['IMG_TSE', 'DESS2TSE']) and knee.path['BMELT'] is None]
        super().__init__(kds, img_size=img_size)
        
    def __getitem__(self, ndx):
        slc = super().__getitem__(ndx)
        ret = {}
        ret['gt_image'] = slc['IMG_TSE']
        ret['cond_image'] = slc['DESS2TSE']
        # ret['path'] = f"knee{slc['knee_ndx']:04d}_slc{slc['slc_ndx']:02d}.png"
        ret['path'] = f"{slc['id']}.png"
        ret['id'] = slc['id']
        return ret

class InpaintTSEDataset(SliceDataset):
    def __init__(self, img_size=256, mask_config={}, **kwargs):
        kds = KneeDataset()
        kds.knees = [knee for knee in kds.knees if all(knee.path[k] for k in ['IMG_TSE', 'DESS2TSE'])][:5]
        super().__init__(kds, img_size=img_size)

        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = (img_size, img_size)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)
    
    def __getitem__(self, ndx):
        slc = super().__getitem__(ndx)
        ret = {}
        img = slc['IMG_TSE'] * 2 - 1
        mask = self.get_mask()
        # mask = torch.zeros([1, *self.image_size])
        # mask[self.image_size[0]//4:self.image_size[0]*3//4,
        #      self.image_size[1]//4:self.image_size[1]*3//4] = 1
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = f"knee{slc['knee_ndx']:04d}_slc{slc['slc_ndx']:02d}.png"
        return ret
    
ROOT = '/mnt/rstor/CSE_CSDS_VXC204/asy51/data/fastmri/'
ANNOT = '/mnt/rstor/CSE_CSDS_VXC204/asy51/data/fastmri_extra/fastmriplus/knee.csv'
CHECK = '/mnt/rstor/CSE_CSDS_VXC204/asy51/data/fastmri_extra/fastmriplus/knee_file_list.csv'
LESION = 'Bone- Subchondral edema'
ACQ = 'CORPDFS_FBK'

class FastSliceDataset(torch.utils.data.Dataset):    
    def __init__(self, img_size=256, n_slc=None, **kwargs):
        self.img_size = (img_size, img_size)
        self.transforms = MT.Compose([
            MT.Lambda(lambda x: einops.rearrange(x, 'd h w -> 1 d h w')),
            MT.Resize(spatial_size=(-1, *self.img_size)),
            MT.SqueezeDim(dim=0),
            # MT.NormalizeIntensityd([q1, q2]),
            MT.ScaleIntensityRangePercentiles(lower=0, upper=99.9, b_min=0, b_max=1, clip=True, relative=False),
            MT.ToTensor(track_meta=False),
        ])

        # h5 path
        self.pathdf = pd.DataFrame({'file': x.split('/')[-1].replace('.h5',''), 'path': x}
                              for x in glob.glob(ROOT + 'multi*/file*.h5'))
        # annotation
        lesions = pd.read_csv(ANNOT, index_col=None, header=0)
        # check
        checked = pd.read_csv(CHECK, index_col=None, header=None, names=['file']).iloc[:-1]
        # train on all slices of healthy knee, healthy slices of nobmel knees
        # first find no bmel knees and totally healthy knees
        lesions_per_knee = lesions.groupby('file')['label'].agg(lambda x: ''.join(x)).reset_index()
        nobmel_knees = lesions_per_knee[~lesions_per_knee['label'].str.contains(LESION)]['file']
        healthy_knees = pd.merge(checked, lesions_per_knee[['file', 'label']], on='file', how='left')
        healthy_knees = healthy_knees[healthy_knees['label'].isna()]['file']
        cand_knees = pd.merge(self.pathdf, pd.merge(healthy_knees, nobmel_knees, how='outer'))
        # second find all slices without any lesions
        cand_knees[['slice', 'acq']] = cand_knees.progress_apply(FastSliceDataset._slc_list, axis=1, result_type='expand')
        cand_knees = cand_knees[cand_knees['acq'] == ACQ].drop(columns=['acq'])
        cand_slices = cand_knees.drop(columns=['path']).explode('slice', ignore_index=True)
        cand_slices['slice'] = cand_slices['slice'].astype(int)
        healthy_slices = pd.merge(cand_slices, lesions[['file', 'slice']], how='outer', indicator=True)
        self.healthy_slices = healthy_slices[healthy_slices['_merge'] == 'left_only'].drop(columns=['_merge'])
        if n_slc is not None:
            self.healthy_slices = self.healthy_slices.sample(n_slc)
        self.img = {f: None for f in self.healthy_slices['file']} # collapses duplicate files into single key
        
    def __len__(self):
        return len(self.healthy_slices)
    
    def __getitem__(self, ndx):
        f, slc_ndx = self.healthy_slices.iloc[ndx]
        if self.img[f] is None:
            filepath = self.pathdf[self.pathdf['file'] == f].iloc[0]['path'] # file should be index?
            hf = h5py.File(filepath)
            self.img[f] = self.transforms(hf['reconstruction_rss'][:][:, ::-1, :].copy())
        img = self.img[f][slc_ndx:slc_ndx+1] * 2 - 1
        ret = {}
        
        mask = bbox2mask(self.img_size,
                         random_bbox(
                             img_shape=self.img_size,
                             max_bbox_shape=(80, 80),
                             max_bbox_delta=60,
                             min_margin=50,
                         )
                        )
        mask = einops.rearrange(torch.from_numpy(mask), 'h w 1 -> 1 h w')
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask
        
        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = f'{f}[{slc_ndx:02}].png'
        return ret
    
    @staticmethod
    def _slc_list(row):
        try:
            hf = h5py.File(row['path'])
            return list(range(hf['reconstruction_rss'].shape[0])), hf.attrs['acquisition']
        except Exception as e:
            print(e)
            return list(), None
        
    # def get_mask(self):
    #     if self.mask_mode == 'bbox':
    #         mask = bbox2mask(self.image_size, random_bbox(
    #                          img_shape=self.img_size,
    #                          max_bbox_shape=(100, 100),
    #                          max_bbox_delta=50,
    #                          min_margin=50,
    #                      ))
    #     elif self.mask_mode == 'center':
    #         h, w = self.image_size
    #         mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
    #     elif self.mask_mode == 'irregular':
    #         mask = get_irregular_mask(self.image_size)
    #     elif self.mask_mode == 'free_form':
    #         mask = brush_stroke_mask(self.image_size)
    #     elif self.mask_mode == 'hybrid':
    #         regular_mask = bbox2mask(self.image_size, random_bbox())
    #         irregular_mask = brush_stroke_mask(self.image_size, )
    #         mask = regular_mask | irregular_mask
    #     elif self.mask_mode == 'file':
    #         pass
    #     else:
    #         raise NotImplementedError(
    #             f'Mask mode {self.mask_mode} has not been implemented.')
    #     return torch.from_numpy(mask).permute(2,0,1)