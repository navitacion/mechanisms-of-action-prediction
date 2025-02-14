import os
import pandas as pd
import hydra
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from sklearn.feature_selection import VarianceThreshold

from src.dataset import MoADataset
from src.model import TablarNet, TablarNet_res

pd.set_option('display.max_rows', None)


@hydra.main('config.yml')
def main(cfg: DictConfig):
    cur_dir = hydra.utils.get_original_cwd()
    os.chdir(cur_dir)
    data_dir = './input'
    train_target = pd.read_csv(os.path.join(data_dir, 'train_targets_scored.csv'))
    train_feature = pd.read_csv(os.path.join(data_dir, 'train_features.csv'))
    df = pd.merge(train_target, train_feature, on='sig_id')
    target_cols = [c for c in train_target.columns if c != 'sig_id']
    feature_cols = [c for c in train_feature.columns if c not in ['sig_id', 'cp_type', 'cp_time', 'cp_dose']]

    print(df.shape)
    df = pd.get_dummies(df, columns=['cp_time','cp_dose'])
    print(df.shape)
    print(type(df))

    # for c in feature_cols:
    #     print('{}: {:.4f}'.format(c, df[c].var()))


    print(df.shape)
    var_thresh = VarianceThreshold(threshold=0.5)
    df = var_thresh.fit_transform(df[feature_cols])
    print(type(df))


    # feature_cols = [c for c in df.columns if c not in target_cols + ['sig_id']]
    #
    # dataset = MoADataset(df, feature_cols, target_cols)
    # dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    #
    # cont_f, cat_f, target = next(iter(dataloader))
    #
    # in_features = 875 + 29 + 4 - 3
    # print(in_features)
    # emb_dims = [(2, 15), (3, 20), (2, 15)]
    # net = TablarNet_res(emb_dims, cfg, in_cont_features=in_features, out_features=206)
    #
    # out = net(cont_f, cat_f)
    # print(out.size())
    # criterion = nn.BCEWithLogitsLoss()
    # loss = criterion(out, target)
    # print(loss)



import torch

def check():

    a = torch.randn((3, 224, 224))

    print(torch.cuda.is_available())

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    a = a.to(device)

    print(a)



if __name__ == '__main__':
    # main()
    check()
