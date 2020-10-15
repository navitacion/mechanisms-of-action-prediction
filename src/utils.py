import random
import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
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


    def _variancethreshold(self, df, threshold=0.5):
        targets = df[self.target_cols]
        var_thresh = VarianceThreshold(threshold=threshold)
        cols = [c for c in df.columns if c not in self.target_cols + ['sig_id', 'is_train', 'cp_type', 'cp_time', 'cp_dose']]
        temp = var_thresh.fit_transform(df[cols])

        out = df[['sig_id', 'is_train', 'cp_type']]
        temp = pd.DataFrame(temp)
        out = pd.concat([out, temp, targets], axis=1)

        return out


    def load_data(self):
        train_target = pd.read_csv(os.path.join(self.data_dir, 'train_targets_scored.csv'))
        train_feature = pd.read_csv(os.path.join(self.data_dir, 'train_features.csv'))
        test = pd.read_csv(os.path.join(self.data_dir, 'test_features.csv'))

        train = pd.merge(train_target, train_feature, on='sig_id')
        self.target_cols = [c for c in train_target.columns if c != 'sig_id']

        test['is_train'] = 0
        train['is_train'] = 1
        df = pd.concat([train, test], axis=0, ignore_index=True)

        # "ctl_vehicle" is not in scope prediction
        df = df[df['cp_type'] != "ctl_vehicle"].reset_index(drop=True)

        return df

    def preprocessing(self, df):
        df = self._get_dummies(df)
        df = self._add_PCA(df, self.cfg)
        if self.cfg.train.var_thresh is not None:
            df = self._variancethreshold(df, self.cfg.train.var_thresh)
        self.feature_cols = [c for c in df.columns if c not in self.target_cols + ['sig_id', 'is_train', 'cp_type', 'cp_time', 'cp_dose']]

        return df

    def split_data(self, df):
        # Split Train, Test
        trainval = df[df['is_train'] == 1].reset_index(drop=True)
        test = df[df['is_train'] == 0].reset_index(drop=True)

        # Split Train, Validation
        trainval['fold'] = -1
        for i, (trn_idx, val_idx) in enumerate(self.cv.split(trainval, trainval[self.target_cols])):
            trainval.loc[val_idx, 'fold'] = i

        return trainval, test



    def run(self):
        # Data Loading
        df = self.load_data()
        # Preprocessing
        df = self.preprocessing(df)
        # Split Data
        trainval, test = self.split_data(df)

        return trainval, test



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



def check_oof_score(cfg, data_dir='./output', target_dir='./input'):

    csv_path = glob.glob(os.path.join(data_dir, f'oof_{cfg.exp.exp_name}*'))
    oof = pd.DataFrame()

    for path in csv_path:
        temp = pd.read_csv(path)
        oof = pd.concat([oof, temp], axis=0)

    oof = oof.sort_values(by='sig_id', ascending=True)

    org_target = pd.read_csv(os.path.join(target_dir, 'train_targets_scored.csv'))
    org_feature = pd.read_csv(os.path.join(target_dir, 'train_features.csv'))
    target_cols = org_target.columns[1:]
    org = pd.merge(org_target, org_feature, on='sig_id')
    org = org[org['cp_type'] != "ctl_vehicle"].reset_index(drop=True)

    org = org.sort_values(by='sig_id', ascending=True)

    score = 0
    for c in target_cols:
        score += log_loss(y_true=org[c].values, y_pred=oof[c].values) / len(target_cols)

    return score
