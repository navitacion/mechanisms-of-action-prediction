import random
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch
from torch.utils.data import DataLoader

from .dataset import MoADataset

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



class Datapreprocessing:
    def __init__(self, data_dir, cfg, cv):
        self.data_dir = data_dir
        self.cfg = cfg
        self.cv = cv

    def _get_dummies(self, df):
        df = pd.get_dummies(df, columns=['cp_time','cp_dose'])
        return df

    def _add_PCA(self, df, cfg):
        # g-features
        g_cols = [c for c in df.columns if 'g-' in c]
        temp = PCA(n_components=cfg.train.g_comp, random_state=cfg.train.seed).fit_transform(df[g_cols])
        temp = pd.DataFrame(temp, columns=[f'g-pca_{i}' for i in range(cfg.train.g_comp)])
        df = pd.concat([df, temp], axis=1)

        # c-features
        c_cols = [c for c in df.columns if 'c-' in c]
        temp = PCA(n_components=cfg.train.c_comp, random_state=cfg.train.seed).fit_transform(df[c_cols])
        temp = pd.DataFrame(temp, columns=[f'c-pca_{i}' for i in range(cfg.train.c_comp)])
        df = pd.concat([df, temp], axis=1)

        del temp

        return df

    def load_data(self):
        train_target = pd.read_csv(os.path.join(self.data_dir, 'train_targets_scored.csv'))
        train_feature = pd.read_csv(os.path.join(self.data_dir, 'train_features.csv'))
        test = pd.read_csv(os.path.join(self.data_dir, 'test_features.csv'))

        train = pd.merge(train_target, train_feature, on='sig_id')
        target_cols = [c for c in train_target.columns if c != 'sig_id']

        test['is_train'] = 0
        train['is_train'] = 1
        df = pd.concat([train, test], axis=0, ignore_index=True)

        # "ctl_vehicle" is not in scope prediction
        df = df[df['cp_type'] != "ctl_vehicle"].reset_index(drop=True)

        return df, target_cols

    def preprocessing(self, df, target_cols):
        df = self._get_dummies(df)
        df = self._add_PCA(df, self.cfg)
        feature_cols = [c for c in df.columns if c not in target_cols + ['sig_id', 'is_train', 'cp_type', 'cp_time', 'cp_dose']]

        return df, feature_cols

    def split_data(self, df, target_cols):
        # Split Train, Test
        trainval = df[df['is_train'] == 1].reset_index(drop=True)
        test = df[df['is_train'] == 0].reset_index(drop=True)

        # Split Train, Validation
        trainval['fold'] = -1
        for i, (trn_idx, val_idx) in enumerate(self.cv.split(trainval, trainval[target_cols])):
            trainval.loc[val_idx, 'fold'] = i

        return trainval, test



    def run(self):
        # Data Loading
        df, target_cols = self.load_data()
        # Preprocessing
        df, feature_cols = self.preprocessing(df, target_cols)
        # Split Data
        trainval, test = self.split_data(df, target_cols)

        return trainval, test, feature_cols, target_cols



def get_dataloaders(trainval, test, batch_size, fold, feature_cols, target_cols):
    # Split Train, Validation
    train = trainval[trainval['fold'] != fold].reset_index(drop=True)
    val = trainval[trainval['fold'] == fold].reset_index(drop=True)

    # Create Dataset
    train_dataset = MoADataset(train, feature_cols, target_cols, phase='train')
    val_dataset = MoADataset(val, feature_cols, target_cols, phase='val')
    test_dataset = MoADataset(test, feature_cols, target_cols, phase='test')

    # Create DataLoaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    }

    return dataloaders



