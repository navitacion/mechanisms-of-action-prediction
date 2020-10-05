import os
import pandas as pd
from torch.utils.data import DataLoader

from src.utils import Encode
from src.dataset import MoADataset_2
from src.model import MyNet

data_dir = './input'
train_target = pd.read_csv(os.path.join(data_dir, 'train_targets_scored.csv'))
train_feature = pd.read_csv(os.path.join(data_dir, 'train_features.csv'))
df = pd.merge(train_target, train_feature, on='sig_id')
target_cols = [c for c in train_target.columns if c != 'sig_id']
feature_cols = [c for c in train_feature.columns if c != 'sig_id']
df = Encode(df)

dataset = MoADataset_2(df, feature_cols, target_cols)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

g, c, cat_f, target = next(iter(dataloader))

print(g.size())
print(c.size())
print(cat_f.size())

emb_dims = [(2, 15), (3, 20), (2, 15)]
net = MyNet(emb_dims, dropout_rate=0.4)

a = net(g, c, cat_f)
