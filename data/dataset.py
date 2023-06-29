import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np
import random

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

from ptoa.data.knee_monai import SliceDataset, KneeDataset
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


class DESS2TSEDataset(SliceDataset):
    def __init__(self, config, kds=None, bmel=False, count=None, **kwargs):
        if kds is None:
            kds = KneeDataset()
            kds.knees = [knee for knee in kds.knees if all(knee.path[k] for k in ['IMG_TSE', 'DESS2TSE', 'BONE_TSE'])]
            if bmel:
                kds.knees = [knee for knee in kds.knees if knee.path['BMELT'] is not None]
            else:
                kds.knees = [knee for knee in kds.knees if knee.path['BMELT'] is None]
            if count:
                random.shuffle(kds.knees)
                kds.knees = kds.knees[:count]
        super().__init__(kds, img_size=config['img_size'], bonemask=config['task'] == 'bone_premask', **kwargs)
        
    def __getitem__(self, ndx):
        slc = super().__getitem__(ndx)
        ret = {}
        ret['tse'] = slc['IMG_TSE']
        ret['dess'] = slc['DESS2TSE']
        ret['bone'] = slc['BONE_TSE']
        ret['bone_nodil'] = slc['BONE_TSE_nodil']
        ret['id'] = slc['id']
        return ret
    

class InpaintBoneDataset(SliceDataset):
    def __init__(self, config, kds=None, bmel=False, count=None, **kwargs):
        if kds is None:
            kds = KneeDataset()
            kds.knees = [knee for knee in kds.knees if all(knee.path[k] for k in ['IMG_TSE', 'DESS2TSE', 'BONE_TSE'])]
            if bmel:
                kds.knees = [knee for knee in kds.knees if knee.path['BMELT'] is not None]
            else:
                kds.knees = [knee for knee in kds.knees if knee.path['BMELT'] is None]
            if count:
                random.shuffle(kds.knees)
                kds.knees = kds.knees[:count]
        super().__init__(kds, img_size=config['img_size'], bonemask=config['task'] == 'bone_premask')
        
    def __getitem__(self, ndx):
        slc = super().__getitem__(ndx)
        ret = {}
        ret['tse'] = slc['IMG_TSE']
        ret['dess'] = slc['DESS2TSE']
        ret['bone'] = slc['BONE_TSE']
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
    
