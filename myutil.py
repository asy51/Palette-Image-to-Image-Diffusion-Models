import ast
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

def load_opts(name, **kwargs):
    f = open(f'./checkpoints/{name}/train_opt.txt', 'r')
    opt = [l.strip().split()[:2] for l in f][1:-1]
    opt = [o + [''] if len(o) == 1  else o for o in opt]
    for o in opt:
        o[0] = o[0][:-1]
        if o[0] == 'gpu_ids':
            o[1] = [int(str_id) for str_id in o[1].split(',') if int(str_id) >= 0]
        else:
            try:
                o[1] = ast.literal_eval(o[1]) # eval ints, floats, bools
            except:
                pass
    opt = {o[0]:o[1] for o in opt}
    opt = argparse.Namespace(**opt)
    for k,v in kwargs.items():
        setattr(opt, k, v)
    return opt

### PLOT FNS

R,G,B = 0,1,2

def make_rgb(img):
    out = np.zeros([3] + list(img.shape))
    out[:,:,:] = img
    return out

def add_mask(img, mask, c='r', alpha=0.2):
    if c == 'r': # RED
        rgb = [R]
    elif c == 'g': # GREEN
        rgb = [G]
    elif c == 'b': # BLUE
        rgb = [B]
    elif c == 'c': # CYAN
        rgb = [G,B]
    elif c == 'm': # MAGENTA
        rgb = [B,R]
    elif c == 'y': # YELLOW
        rgb = [R,G]
    else:
        raise ValueError('c must be one of [r,g,b,c,m,y]')
    img[rgb,:,:] += (mask * alpha)
    return img

def scale(img, min_q=0., max_q=1., new_min=None, new_max=None):
    min_val = np.quantile(img, min_q)
    max_val = np.quantile(img, max_q)
    if new_min is None:
        new_min = min_val
    if new_max is None:
        new_max = max_val
    img = np.clip(img, min_val, max_val)
    img = img / (max_val - min_val) * (new_max - new_min)
    img = img - img.min() + new_min
    return img

def subplots(cols=1, figsize=7.5):
    fig, ax = plt.subplots(1, cols, figsize=(cols * figsize, figsize))
    return fig, ax

def plots(imgs, cp=None, title=None, axtitles=None, cbar=True, figsize=7.5, reorient=True, axis=True):
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
    n_axes = len(imgs)
    if axtitles is None:
        axtitles = [None] * n_axes
    elif not isinstance(axtitles, (list, tuple)):
        axtitles = [axtitles]

    axis = 'on' if axis else 'off'

    fig, ax = subplots(n_axes, figsize)
    if n_axes == 1:
        ax = [ax]
    fig.suptitle(title)
    # fig.tight_layout()
    if cp is not None and reorient:
        cp = cp[1], cp[0]
    for i, img in enumerate(imgs):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
        if len(img.shape) == 3: # RGB
            if reorient:
                img = np.moveaxis(img, 0, -1)
            cmap = None
        else:
            cmap = 'gray'
        if reorient:
            img = np.fliplr(np.rot90(img, k=3))
        mat = ax[i].matshow(img, cmap=cmap)
        if cp is not None:
            ax[i].scatter(cp[0], cp[1], s=2_000, edgecolors='r', alpha=0.5, facecolors='none', linewidths=1.25)
        ax[i].set_title(axtitles[i]) #, fontdict={'size': 24})
        if cbar:
            plt.colorbar(mat, ax=ax[i])
        
        ax[i].axis(axis)
    return fig, ax

def rot(x: torch.Tensor, dims=(-2, -1)):
    return x.fliplr().rot90(1, dims=dims)

def path2base(path):
    """
    given LiX path, return home path
    """
    split = path.split('/')
    values = ['moon' if 'MOON' in split[2] else 'comet'] + split[3:8] + split[-1:]
    # labels = ['study', 'control', 'site', 'id', 'date', 'knee', 'suffix']
    # ret = {labels[i]: values[i] for i in range(len(labels))}
    return f"{'-'.join(values[:-1])}"

# dice_fn = DiceMetric(reduction='mean')
def dice_fn(a,b):
    a = a.to(bool) if isinstance(a, torch.Tensor) else a.astype(bool)
    b = b.to(bool) if isinstance(b, torch.Tensor) else b.astype(bool)
    return 2 * (a & b).sum() / (a.sum() + b.sum())

def scale(img, min_q=0., max_q=1., new_min=None, new_max=None):
    min_val = np.quantile(img, min_q)
    max_val = np.quantile(img, max_q)
    if new_min is None:
        new_min = min_val
    if new_max is None:
        new_max = max_val
    img = np.clip(img, min_val, max_val)
    img = img / (max_val - min_val) * (new_max - new_min)
    img = img - img.min() + new_min
    return img

def center_of_mass(a):
    return (a * np.mgrid[0:a.shape[0], 0:a.shape[1]]).sum(1).sum(1)/a.sum()

def roi(arr, cp, sz):
    """
    Create a binary mask for a 2D array with a square ROI centered at cp set to 1.
    
    Parameters:
    - arr: np.ndarray, the input 2D array.
    - cp: tuple, the (y, x) coordinates of the center point.
    - sz: int, the length of each side of the square ROI.
    
    Returns:
    - mask: np.ndarray, a binary mask with the ROI set to 1, the same shape as arr.
    """
    # Initialize the mask with zeros
    mask = np.zeros_like(arr, dtype=bool)
    
    # Calculate half size to manage even size
    half_sz = sz // 2
    
    # Determine the start and end points of the ROI
    start_y = int(max(cp[0] - half_sz, 0))
    end_y = int(min(cp[0] + half_sz + (sz % 2), arr.shape[0]))
    start_x = int(max(cp[1] - half_sz, 0))
    end_x = int(min(cp[1] + half_sz + (sz % 2), arr.shape[1]))
    
    # Set the ROI region to 1
    mask[start_y:end_y, start_x:end_x] = 1
    
    return mask

def roi_combined(bmel, sz=96):
    mask_roi_combined = np.zeros_like(bmel, dtype=bool)
    for bmel_ndx in np.unique(bmel):
        if bmel_ndx == 0: continue
        bmel_current = bmel == bmel_ndx
        com = center_of_mass(bmel_current)
        mask_roi = roi(bmel, com, sz=sz)
        mask_roi_combined = mask_roi_combined | mask_roi
    return mask_roi_combined