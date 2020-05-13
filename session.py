import os
import time
from collections import UserDict

import numpy as np
import torch
import re

from utils import dict_update


class Sess(UserDict):
    def __init__(self, d):
        """
        :param kwargs: keyword arguments to pass to the session constructor. {path, batch_size, samples, D, G, epoch, batch, ...}
        """
        super().__init__(d)

        self.defaults = {
            'batch_size': 64,
            'path': f'./runs',

            # keep track of loss and generated, "fake" samples
            'samples': [],
            'D': {
                'state_dict': {},
                'optimizer_state_dict': {},
                'loss': 0,
                'losses': [],
            },
            'G': {
                'state_dict': {},
                'optimizer_state_dict': {},
                'loss': 0,
                'losses': [],
            },

            'epoch': 0,  # completed epochs
            'batch': 0,  # completed batches

            'alltime_start_time': time.time(),
        }

        dict_update(self, dict_update(self.defaults, self))

    def without_keys(self, keys):
        return {k: v for k, v in self.items() if k not in keys}

    def load(self, ckpt=None, **kwargs):
        if ckpt is False:
            return self

        path = self.get('path', self.defaults.get('path'))
        ckpt_dir = os.path.join(path, 'checkpoints')

        # first file to exist
        ckpt, ckpt_path = load_ckpt(ckpt_dir, ckpt=ckpt, ret_path=True, **kwargs)
        dict_update(self, dict_update(self.defaults, dict_update(self, ckpt)))
        del ckpt
        print(f'Session loaded from "{ckpt_path}"')

        return self


def get_filenames_number_dict(dirname, pattern="(\d+).", reverse=True):
    from collections import OrderedDict
    files = [(extract_number(f, pattern), os.path.join(dirname, f)) for f in os.listdir(dirname)]
    return OrderedDict(sorted(files, key=lambda kv: kv[0], reverse=reverse))


#FIXME: not working
def extract_number(f, pattern="(\d+)."):
    s = re.findall(pattern, f)
    return int(s[0]) if s else -1, f


# move this to session.py
# FIXME: trying to load specific checkpoints but they don't work
def load_ckpt(ckpt_dir=f'runs/untitled/checkpoints', pattern=r'checkpoint_(\d+).ckpt', ckpt=-1, ret_path=False):
    """tries to load and if a ckpt is corrupted it'll get the latest valid one before it
    :param pattern:
    :param ckpt_dir:
    :param ckpt: (dict|int|str): a specific checkpoint to load, either a checkpoint itself (in that case it will be returned),
        or an index (can be negative), or str as filepath, or str indicating the number of the checkpoint
    :param ret_path: bool: if true, returns a tuple (ckpt, ckpt_path)

    >> load_ckpt(ckpt=-1) # will load the latest file (index -1) (default)
    >> load_ckpt(ckpt="69") # will load the file "checkpoint_69.ckpt"
    >> load_ckpt(ckpt="checkpoint_69.ckpt") # will load the file "checkpoint_69.ckpt"
    >> load_ckpt(ckpt="full/path/to/checkpoint_69.ckpt") # will load the file "checkpoint_69.ckpt"
    """

    if type(ckpt) is dict:
        return ckpt

    fnames_dict = get_filenames_number_dict(ckpt_dir, pattern=pattern)

    if type(ckpt) is str:  # path
        if not os.path.exists(ckpt):
            ckpt = os.path.join(ckpt_dir, ckpt)

    if type(ckpt) is int:  # index
        ids = list(zip(*list(fnames_dict.keys())))
        if ids:
            ids = ids[0]
            ckpt = str(ids[np.argsort(ids)[ckpt]])
    
    print('ckpt target:',ckpt)
    
    for i, ((ckpt_id, ckpt_name), ckpt_path) in enumerate(fnames_dict.items()):
        if ckpt == str(ckpt_id) or ckpt == ckpt_path:
            try:
                ckpt_ = torch.load(ckpt_path)
                print(f'Loaded {i}th checkpoint: {ckpt_path}')
                return (ckpt_, ckpt_path) if ret_path else ckpt_
            except RuntimeError as e:
                if 'unexpected EOF' in str(e):
                    print(f'Checkpoint file corrupted: "{os.path.split(ckpt_path)[-1]}": "{e}", deleting...', end='')
                    os.remove(ckpt_path)
                    print('deleted')
                else:
                    raise e
#         else:
#             print(f'NOPE: {ckpt} =!= {ckpt_id}')

    print(f"No valid checkpoints found in {ckpt_dir}")
    return (None, '') if ret_path else None


# print('testing loading session')
# ckpt = load_ckpt("../../../../datasets/anime-faces/")
# print('ckpt:', ckpt)
