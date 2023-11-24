import glob
import os
import random

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn


def get_assigned_file(checkpoint_dir, num):
    assign_file = os.path.join(checkpoint_dir, f"{num}.tar")
    return assign_file


def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist = [x for x in filelist if os.path.basename(x) != 'best_model.tar']
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file


def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)


def get_latest_dir(checkpoint_dir):
    # checkpoint_dir has a bunch of directories with names like yyyymmdd_hhmmss, just get the
    # latest one
    dirlist = glob.glob(os.path.join(checkpoint_dir, '*'))
    if len(dirlist) == 0:
        return ValueError("checkpoint dir not found")
    dirlist = sorted(dirlist)
    return dirlist[-1]


def get_model_file(cfg):
    cp_cfg = cfg.checkpoint
    if cp_cfg.time == "latest":
        dir = get_latest_dir(cp_cfg.dir)
    else:
        dir = os.path.join(cp_cfg.dir, cp_cfg.time)

    print(f"Using checkpoint dir: {dir}")
    return get_assigned_file(dir, cp_cfg.test_iter)


def fix_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def model_to_dict(model):
    if isinstance(model, nn.Module):
        model_dict = {}
        children = list(model.named_children())
        if len(children) > 0:
            for name, module in children:
                model_dict[name] = model_to_dict(module)
        else:
            return str(model)
        return model_dict
    else:
        return str(model)


def opt_to_dict(opt):
    opt_dict = opt.param_groups[0].copy()
    opt_dict.pop('params')
    return opt_dict

def hydra_setup():
    os.environ["HYDRA_FULL_ERROR"] = "1"
    try:
        OmegaConf.register_new_resolver("mul", lambda x, y: float(x) * float(y))
    except:
        pass
