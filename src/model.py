import os
import torch
from torch import nn
import pandas as pd
import torch.nn.functional as F



# Tablar Dataを想定したneural-network
# https://yashuseth.blog/2018/07/22/pytorch-neural-network-for-tabular-data-with-categorical-embeddings/
class LinearReluBnDropout(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate):
        super(LinearReluBnDropout, self).__init__()

        self.block = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(in_features, out_features)),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_features),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        x = self.block(x)

        return x


class SimpleDenseNet(nn.Module):
    def __init__(self, cfg, in_features=875, out_features=206):
        super(SimpleDenseNet, self).__init__()
        h = cfg.train.hidden_size
        d = cfg.train.dropout_rate

        self.first_bn = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.2)
        )

        self.block = nn.Sequential(
            LinearReluBnDropout(in_features=in_features, out_features=h, dropout_rate=d),
            LinearReluBnDropout(in_features=h, out_features=h, dropout_rate=d)
        )

        self.last = nn.utils.weight_norm(nn.Linear(h, out_features))

    def forward(self, x):
        x = self.first_bn(x)
        x = self.block(x)
        x = self.last(x)

        return x
