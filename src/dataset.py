from torch.utils.data import Dataset
import torch


class MoADataset(Dataset):
    def __init__(self, features, targets, ids, phase='train'):
        self.features = features
        self.targets = targets
        self.ids = ids
        self.phase = phase


    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        input = torch.tensor(self.features[idx, :], dtype=torch.float)

        if self.phase == 'train':
            target = torch.tensor(self.targets[idx, :], dtype=torch.float)

            return input, target

        elif self.phase == 'val':
            target = torch.tensor(self.targets[idx, :], dtype=torch.float)
            sig_id = self.ids[idx]

            return input, target, sig_id

        else:
            sig_id = self.ids[idx]

            return input, sig_id