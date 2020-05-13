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



def tensorimg_from_np(img):
    return np.transpose(img, (1, 2, 0))

# helper function for viewing a list of passed in sample images
def view_samples(sample, epoch=None, shape=None):
    n_imgs = len(sample)
    ncols = 8
    nrows = int(np.floor(n_imgs/ncols))
    fig, axes = plt.subplots(figsize=(ncols*2, nrows*2), nrows=nrows, ncols=ncols, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), sample):
        img = tensorimg_from_np(unscale(img))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
        if shape is None: # inferring the dimensions
            vol = np.prod(img.shape)
            vol //= 3
            w = int(round(np.sqrt(vol)))
            shape = w, w, 3
        
        im = ax.imshow(img.reshape(shape))
    if epoch is not None:
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
