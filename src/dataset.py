from torch.utils.data import Dataset
import torch

class MoADataset(Dataset):
    def __init__(self, df, feature_cols, target_cols, phase='train'):
        self.df = df
        self.cat_cols = ['cp_type', 'cp_time', 'cp_dose']
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

        if self.phase != 'test':
            target = self.df[self.target_cols].iloc[idx]
            target = torch.tensor(target.values, dtype=torch.float32)

            return cont_f, cat_f, target

        else:
            sig_id = self.df['sig_id'].iloc[idx]
            return cont_f, cat_f, sig_id



class MoADataset_2(Dataset):
    def __init__(self, df, feature_cols, target_cols, phase='train'):
        self.df = df
        self.cat_cols = ['cp_type', 'cp_time', 'cp_dose']
        self.g_cols = [c for c in feature_cols if 'g-' in c]
        self.c_cols = [c for c in feature_cols if 'c-' in c]
        self.target_cols = target_cols
        self.phase = phase


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        g_f = self.df[self.g_cols].iloc[idx]
        g_f = torch.tensor(g_f.values, dtype=torch.float32)

        c_f = self.df[self.c_cols].iloc[idx]
        c_f = torch.tensor(c_f.values, dtype=torch.float32)

        cat_f = self.df[self.cat_cols].iloc[idx]
        cat_f = torch.tensor(cat_f.values, dtype=torch.long)

        if self.phase != 'test':
            target = self.df[self.target_cols].iloc[idx]
            target = torch.tensor(target.values, dtype=torch.float32)

            return g_f, c_f, cat_f, target

        else:
            return g_f, c_f, cat_f

