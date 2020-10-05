import os
import torch
from torch import nn
import pandas as pd

from src.dataset import MoADataset_2
from src.utils import Encode


class DenseModel(nn.Module):
    def __init__(self, cfg, in_features=875, out_features=206):
        super(DenseModel, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(in_features, cfg.train.hidden_size),
            nn.BatchNorm1d(cfg.train.hidden_size),
            nn.Dropout(cfg.train.dropout_rate, inplace=True),
            nn.PReLU(),
            nn.Linear(cfg.train.hidden_size, cfg.train.hidden_size),
            nn.BatchNorm1d(cfg.train.hidden_size),
            nn.Dropout(cfg.train.dropout_rate, inplace=True),
            nn.PReLU()
        )
        self.last = nn.Linear(cfg.train.hidden_size, out_features)

    def forward(self, x):
        x = self.block(x)
        x = self.last(x)

        return x


# Tablar Dataを想定したneural-network
# https://yashuseth.blog/2018/07/22/pytorch-neural-network-for-tabular-data-with-categorical-embeddings/
class LinearReluBnDropout(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate):
        super(LinearReluBnDropout, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_features),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        x = self.block(x)

        return x


class TablarNet(nn.Module):
    def __init__(self, emb_dims, cfg, in_features=875, out_features=206):
        super(TablarNet, self).__init__()

        self.embedding_layer = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        self.dropout = nn.Dropout(cfg.train.dropout_rate, inplace=True)

        self.first_bn_layer = nn.BatchNorm1d(872)

        first_in_feature = in_features - 3 + sum([y for x, y in emb_dims])

        self.block = nn.Sequential(
            LinearReluBnDropout(in_features=first_in_feature,
                                out_features=cfg.train.hidden_size,
                                dropout_rate=cfg.train.dropout_rate),
            LinearReluBnDropout(in_features=cfg.train.hidden_size,
                                out_features=cfg.train.hidden_size,
                                dropout_rate=cfg.train.dropout_rate)
        )

        self.last = nn.Linear(cfg.train.hidden_size, out_features)

    def forward(self, cont_f, cat_f):

        cat_x = [layer(cat_f[:, i]) for i, layer in enumerate(self.embedding_layer)]
        cat_x = torch.cat(cat_x, 1)
        cat_x = self.dropout(cat_x)

        cont_x = self.first_bn_layer(cont_f)

        x = torch.cat([cont_x, cat_x], 1)

        x = self.block(x)
        x = self.last(x)

        return x
