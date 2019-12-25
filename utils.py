import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import unittests as tests
import os

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms



# helper conv function
def conv(in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization"""
    layers = []
    conv_layer = nn.Conv2d(in_channels, out_channels, 
                           kernel_size, stride, padding, bias=not batch_norm)
    
    # append conv layer
    layers.append(conv_layer)

    if batch_norm:
        # append batchnorm layer
        layers.append(nn.BatchNorm2d(out_channels))
     
    # using Sequential container
    return nn.Sequential(*layers)



# helper deconv function
def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    ## create a sequence of transpose + optional batch norm layers
    layers = []
    
    deconv_layer = nn.ConvTranspose2d(in_channels, out_channels,
                                      kernel_size, stride=2, padding=1, bias=not batch_norm)
    layers.append(deconv_layer)
    if batch_norm:
        layers.append(
            nn.BatchNorm2d(out_channels)
        )
    
    # using Sequential container
    return nn.Sequential(*layers)


def weights_init_normal(m, gain=0.02):
    """
    Applies initial weights to certain layers in a model .
    The weights are taken from a normal distribution 
    with mean = 0, std dev = 0.02.
    :param m: A module or layer in a network    
    """
    # classname will be something like:
    # `Conv`, `BatchNorm2d`, `Linear`, etc.
    classname = m.__class__.__name__
    
    # Applying initial weights to convolutional and linear layers
    if 'Conv' in classname or 'Linear' in classname:
        nn.init.xavier_normal_(m.weight, gain=gain)


# helper function for viewing a list of passed in sample images
def view_samples(epoch, sample, shape=None):
    fig, axes = plt.subplots(figsize=(16,4), nrows=2, ncols=8, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), sample):
        img = np.transpose(unscale(img), (1, 2, 0))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
        if shape is None: # inferring the dimensions
            vol = np.prod(img.shape)
            vol //= 3
            w = int(round(np.sqrt(vol)))
            shape = w, w, 3
        
        im = ax.imshow(img.reshape(shape))
    fig.suptitle(f'[Epoch: {epoch}] Generated samples')
    return fig, axes


def scale(x, feature_range=(-1, 1)):
    ''' Scale takes in an image x and returns that image, scaled
       with a feature_range of pixel values from -1 to 1. 
       This function assumes that the input x is already scaled from 0-1.'''
    # assume x is scaled to (0, 1)
    # scale to feature_range and return scaled x
    min_,max_ = feature_range
    x = x * (max_-min_) + min_
    return x

def unscale(img):
    ## given a scaled image (-1,1), turns it back to the normal 0-255
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    return ((img + 1)*255 / (2)).astype(np.uint8)


def elapsed_time(duration):
    hours, rem = divmod(duration, 3600)
    minutes, seconds = divmod(rem, 60)
    hours, minutes = int(hours), int(minutes)
    # return "{:0>2}h:{:0>2}m:{:05.2f}s".format(hours, minutes, seconds)
    if hours == 0 and minutes == 0:
        return "{:05.2f}s".format(seconds)
    elif hours == 0:
        return "{:0>2}m:{:05.0f}s".format(minutes, seconds)
    else:
        return "{:0>2}h:{:0>2}m".format(hours, minutes)



def extract_number(f, pattern="(\d+)."):
    import re
    s = re.findall(pattern, f)
    return (int(s[0]) if s else -1, f)


def get_sorted_filenames_by_number(dirname, pattern="(\d+).", reverse=False):
    return list(
        map(lambda f: os.path.join(dirname, f),
            sorted(os.listdir(dirname), key=extract_number, reverse=reverse)))


#move this to session.py
def load_ckpt(ckpt_dir=f'runs/untitled/checkpoints', pattern=f'ckpt_(\d+).ckpt', ckpt=-1, ret_path=False):
    """tries to load and if a ckpt is corrupted it'll get the latest valid one before it
    :param ckpt: (dict|int|str) a specific checkpoint to load, either a checkpoint itself (in that case it will be returned),
        or an index (can be negative), or a filepath
    :param ret_path: (bool) if true, returns a tuple (ckpt, ckpt_path)
    """
    
    if type(ckpt) is dict:
        return ckpt
    
    fnames = get_sorted_filenames_by_number(ckpt_dir, pattern=pattern, reverse=True)

    if type(ckpt) is str: # path
        if not os.path.exists(ckpt):
            os.path.join(ckpt_dir, ckpt)
        fnames = [ckpt]
    

    if type(ckpt) is int: # index
        index = ckpt
        fnames = np.squeeze([fnames[-index-1::]])
    
    
    for ckpt_path in fnames:
        try:
            ckpt = torch.load(ckpt_path)
            print(f'Checkpoint loaded: {ckpt_path}')
            return (ckpt, ckpt_path) if ret_path else ckpt
        except RuntimeError as e:
            if 'unexpected EOF' in str(e):
                print(f'Checkpoint file corrupted: "{os.path.split(ckpt_path)[-1]}": "{e}", deleting...', end='')
                os.remove(ckpt_path)
                print('deleted')
            else:
                raise e
    
    print(f"No valid checkpoints found in {ckpt_dir}")
    return (None, '') if ret_path else None

def dict_update(d, u):
    import collections.abc
    if u is None:
        u = {}
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
