import random
import os
import numpy as np
import pandas as pd
import torch


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def Encode(df):
    cp_type_encoder = {
        'trt_cp': 0,
        'ctl_vehicle': 1
    }

    cp_time_encoder = {
        48: 1,
        72: 2,
        24: 0
    }

    cp_dose_encoder = {
        'D1': 0,
        'D2': 1
    }

    df['cp_type'] = df['cp_type'].map(cp_type_encoder)
    df['cp_time'] = df['cp_time'].map(cp_time_encoder)
    df['cp_dose'] = df['cp_dose'].map(cp_dose_encoder)

    for c in ['cp_type', 'cp_time', 'cp_dose']:
        df[c] = df[c].astype(int)

    return df