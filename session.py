import pickle as pkl
import json
from collections import UserDict
import time
import os
import torch
import utils
from utils import dict_update


class Sess(UserDict):
    
    def __init__(self,
                 init='sess.sess.json',
                 defaults={
                     'batch_size': 64,
                     'path': f'./runs/untitled'
                 },
                 **kwargs):
        """
        :param sess: old session. dict or path to session dict or None: auto loads '.sess.json' file.
                       To stop this, pass an empty dict
        :param batch_size: 
        :param path: 
        :param kwargs: 
        """
        super().__init__(**kwargs)

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

#         if type(init) is str:
#             self.load(init)
#             init = self
        if type(init) is not dict:
            init = {}
        
#         dict_update(sess, dict_update(defaults, dict_update(sess, ckpt)))
        
        dict_update(self, dict_update(dict_update(self.defaults, defaults), init))
    
    def without_keys(self, keys):
        return {k: v for k, v in self.items() if k not in keys}

    def load(self, ckpt=None) -> bool:
        if ckpt is False:
            return self

        path = self.get('path', self.defaults.get('path'))
        ckpt_dir = os.path.join(path, 'checkpoints')
        
        
#         try:
        # first file to exist
        ckpt, ckpt_path = utils.load_ckpt(ckpt_dir, ckpt=ckpt, ret_path=True)
        dict_update(self, dict_update(self.defaults, dict_update(self, ckpt)))
        del ckpt
        print(f'Session loaded from "{ckpt_path}"')
#         except Exception as e:
#             print(
#                 f"WARNING: couldn't load checkpoint from: \"{ckpt_dir}\": {e}"
#             )
        
        return self


