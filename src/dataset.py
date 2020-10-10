from torch.utils.data import Dataset
import torch

class MoADataset(Dataset):
    def __init__(self, df, feature_cols, target_cols, phase='train'):
        self.df = df
        self.cat_cols = ['cp_time', 'cp_dose']
        self.cont_cols = [c for c in feature_cols if c not in self.cat_cols]
        self.target_cols = target_cols
        self.phase = phase

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        cont_f = self.df[self.cont_cols].iloc[idx]
        cont_f = torch.tensor(cont_f.values, dtype=torch.float32)

        cat_f = self.df[self.cat_cols].iloc[idx]
        cat_f = torch.tensor(cat_f.values, dtype=torch.long)

        if self.phase == 'train':
            target = self.df[self.target_cols].iloc[idx]
            target = torch.tensor(target.values, dtype=torch.float32)

            return cont_f, cat_f, target

        elif self.phase == 'val':
            target = self.df[self.target_cols].iloc[idx]
            target = torch.tensor(target.values, dtype=torch.float32)
            sig_id = self.df['sig_id'].iloc[idx]

            return cont_f, cat_f, target, sig_id

        else:
            sig_id = self.df['sig_id'].iloc[idx]
            return cont_f, cat_f, sig_id



class MoADataset_2(Dataset):
    def __init__(self, df, feature_cols, target_cols, phase='train'):
        self.df = df
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.phase = phase


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input = self.df[self.feature_cols].iloc[idx]
        input = torch.tensor(input.values, dtype=torch.float32)

        if self.phase == 'train':
            target = self.df[self.target_cols].iloc[idx]
            target = torch.tensor(target.values, dtype=torch.float32)

            return input, target

        elif self.phase == 'val':
            target = self.df[self.target_cols].iloc[idx]
            target = torch.tensor(target.values, dtype=torch.float32)
            sig_id = self.df['sig_id'].iloc[idx]

            return input, target, sig_id

        else:
            sig_id = self.df['sig_id'].iloc[idx]
            return input, sig_id

