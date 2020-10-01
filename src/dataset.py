from torch.utils.data import DataLoader, Dataset
import torch
import os
import pandas as pd
from category_encoders import OrdinalEncoder


class MoADataset(Dataset):
    def __init__(self, df, feature_cols, target_cols, phase='train'):
        self.df = df
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.phase = phase

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        feature = self.df[self.feature_cols].iloc[idx]
        feature = torch.tensor(feature.values, dtype=torch.float32)
        if self.phase != 'test':
            target = self.df[self.target_cols].iloc[idx]
            target = torch.tensor(target.values, dtype=torch.float32)

            return feature, target

        else:
            return feature



if __name__ == '__main__':
    data_dir = '../input'
    train_target = pd.read_csv(os.path.join(data_dir, 'train_targets_scored.csv'))
    train_feature = pd.read_csv(os.path.join(data_dir, 'train_features.csv'))

    test = pd.read_csv(os.path.join(data_dir, 'test_features.csv'))
    train = pd.merge(train_target, train_feature, on='sig_id')
    target_cols = [c for c in train_target.columns if c != 'sig_id']
    feature_cols = [c for c in train_feature.columns if c != 'sig_id']

    test['is_train'] = 0
    train['is_train'] = 1
    df = pd.concat([train, test], axis=0, ignore_index=True)

    # Label Encoder
    cols = ['cp_type', 'cp_time', 'cp_dose']
    encoder = OrdinalEncoder(cols=cols)
    df = encoder.fit_transform(df)

    train = df[df['is_train'] == 1]

    dataset = MoADataset(train, feature_cols, target_cols)

    f, t = dataset.__getitem__(8)

    print(f.size())
    print(f)
    print(t.size())
    print(t)
    print(dataset.__len__())
