from IPython import embed
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

from ptoa.data.knee_monai import SliceDataset, KneeDataset
from ptoa.data.fastmri_dataset import FastSliceDataset, FastTranslateDataset, CacheFastSliceDataset
import monai.transforms as MT

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


class MoonCometTranslateDataset(SliceDataset):
    def __init__(self, img_size=320, **kwargs):
        kds = KneeDataset()
        kds.knees = [knee for knee in kds.knees if all(knee.path[k] for k in ['IMG_TSE', 'DESS2TSE'])]
        super().__init__(kds, img_size=img_size)
        
    def __getitem__(self, ndx):
        slc = super().__getitem__(ndx)
        ret = {}
        ret['gt_image'] = slc['IMG_TSE'] * 2 - 1
        ret['cond_image'] = slc['DESS2TSE'] * 2 - 1
        ret['path'] = f"knee{slc['base']}-slc{slc['slc_ndx']:02d}.png"
        return ret

class MoonCometInpaintDataset(SliceDataset):
    def __init__(self, img_size=320, **kwargs):
        kds = KneeDataset()
        kds.knees = [knee for knee in kds.knees if all(knee.path[k] for k in ['IMG_TSE'])]
        super().__init__(kds, img_size=img_size)

    def __getitem__(self, ndx):
        slc = super().__getitem__(ndx)
        img = slc['IMG_TSE'] * 2 - 1
        mask = torch.from_numpy(bbox2mask(img.shape[-2:],
                    random_bbox(
                        img_shape=img.shape[-2:],
                        max_bbox_shape=(80, 80),
                        max_bbox_delta=60,
                        min_margin=50,
                    )
                )).to(torch.uint8).squeeze(-1).unsqueeze(0)
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret = {
            'gt_image': img,
            'cond_image': cond_image,
            'mask_image': mask_img,
            'mask': mask,
            'path': f"knee{slc['base']}-slc{slc['slc_ndx']:02d}.png",
        }

        return ret

class FastInpaintDataset(FastSliceDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.df = self.df[(self.df['acquisition'] == 'CORPDFS_FBK') & (self.df['split'] == 'train') & (self.df['label'] != 'Bone- Subchondral edema')]
        self.df = self.df[self.df['label'].isna()]

    def __getitem__(self, ndx):
        slc = super().__getitem__(ndx)
        ret_row = self.df.iloc[ndx]

        img = slc['img'] * 2 - 1
        mask = torch.from_numpy(bbox2mask(img.shape[-2:],
                    random_bbox(
                        img_shape=img.shape[-2:],
                        max_bbox_shape=(80, 80),
                        max_bbox_delta=60,
                        min_margin=50,
                    )
                )).to(torch.uint8).squeeze(-1).unsqueeze(0)
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret = {
            'gt_image': img,
            'cond_image': cond_image,
            'mask_image': mask_img,
            'mask': mask,
            'path': f"{ret_row['file']}-{int(float(ret_row['slice']))}.png",
        }

        return ret
    
class FastTranslateDataset(FastTranslateDataset):
    def __getitem__(self, ndx):
        slc = super().__getitem__(ndx)
        ret_row = self.df.iloc[ndx]

        ret = {
            'gt_image': slc['img_fs'] * 2 - 1,
            'cond_image': slc['img_nofs'] * 2 - 1,
            'path': f"{ret_row['file_fs']}-{int(float(ret_row['slice']))}.png",
        }

        return ret
    

# class InpaintTSEDataset(SliceDataset):
#     def __init__(self, slices=None, knees=None, img_size=256, mask_config={'mask_mode': 'bone'}, **kwargs):
#         if slices is not None:
#             self.slices = slices
#         else:
#             if knees is None:
#                 kds = KneeDataset()
#                 kds.knees = [k for k in kds.knees if k.base in clean_nobmel_knees]

#                 kds.knees = [knee for knee in kds.knees if all(knee.path[k] for k in ['IMG_TSE', 'DESS2TSE', 'BONE_TSE'])]
#                 knees = kds.knees
#             super().__init__(knees, img_size=img_size)

#         self.mask_config = mask_config
#         self.mask_mode = self.mask_config['mask_mode']
#         self.image_size = (img_size, img_size)

#     def get_mask(self):
#         if self.mask_mode == 'bbox':
#             mask = bbox2mask(self.image_size, random_bbox())
#         elif self.mask_mode == 'center':
#             h, w = self.image_size
#             mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
#         elif self.mask_mode == 'irregular':
#             mask = get_irregular_mask(self.image_size)
#         elif self.mask_mode == 'free_form':
#             mask = brush_stroke_mask(self.image_size)
#         elif self.mask_mode == 'hybrid':
#             regular_mask = bbox2mask(self.image_size, random_bbox())
#             irregular_mask = brush_stroke_mask(self.image_size, )
#             mask = regular_mask | irregular_mask
#         elif self.mask_mode == 'file':
#             pass
#         else:
#             raise NotImplementedError(
#                 f'Mask mode {self.mask_mode} has not been implemented.')
#         return torch.from_numpy(mask).permute(2,0,1)
    
#     def __getitem__(self, ndx):
#         slc = super().__getitem__(ndx)
#         ret = {}
#         img = slc['IMG_TSE'] * 2 - 1
#         if self.mask_mode == 'bone':
#             mask = (slc['BONE_TSE'] > 0).to(torch.uint8)
#         else:
#             mask = self.get_mask()
#         cond_image = img*(1. - mask) + mask*torch.randn_like(img)
#         mask_img = img*(1. - mask) + mask

#         ret['gt_image'] = img
#         ret['cond_image'] = cond_image
#         ret['mask_image'] = mask_img
#         ret['mask'] = mask
#         ret['id'] = slc['id']
#         ret['path'] = f"{slc['id']}.png"
#         return ret