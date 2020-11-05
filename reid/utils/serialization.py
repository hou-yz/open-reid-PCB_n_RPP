from __future__ import print_function, absolute_import

import json
import os
import os.path as osp

import torch
from torch.nn import Parameter


def read_json(fpath):
    """
    Reads a file.

    Args:
        fpath: (str): write your description
    """
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """
    Write the given object to a json file.

    Args:
        obj: (todo): write your description
        fpath: (str): write your description
    """
    os.makedirs(osp.dirname(fpath), exist_ok=True)
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    """
    Save checkpoint to file.

    Args:
        state: (todo): write your description
        is_best: (bool): write your description
        fpath: (str): write your description
    """
    os.makedirs(osp.dirname(fpath), exist_ok=True)
    if int(state['epoch']) % 10 == 0:
        torch.save(state, fpath)
    if is_best:
        torch.save(state, osp.join(osp.dirname(fpath), 'model_best.pth.tar'))


def load_checkpoint(fpath):
    """
    Load checkpoint from file.

    Args:
        fpath: (str): write your description
    """
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath)
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))


def copy_state_dict(state_dict, model, strip=None):
    """
    Copy a dict of model parameters to a dict.

    Args:
        state_dict: (dict): write your description
        model: (todo): write your description
        strip: (str): write your description
    """
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    return model
