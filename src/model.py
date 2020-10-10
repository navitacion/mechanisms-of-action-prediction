import os
import torch
from torch import nn
import pandas as pd



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


class ResidualUnit(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate):
        super(ResidualUnit, self).__init__()

        self.block = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(in_features, out_features)),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(out_features, out_features)),
        )

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = lambda x: x

    def forward(self, x):
        out1 = self.block(x)
        out2 = self.shortcut(x)
        x = self.relu(out1 + out2)
        x = self.dropout(x)
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


class TablarNet(nn.Module):
    def __init__(self, emb_dims, cfg, in_cont_features=875, out_features=206):
        super(TablarNet, self).__init__()

        self.embedding_layer = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        self.dropout = nn.Dropout(cfg.train.dropout_rate, inplace=True)

        self.first_bn_layer = nn.Sequential(
            nn.BatchNorm1d(in_cont_features),
            nn.Dropout(cfg.train.dropout_rate)
        )

        first_in_feature = in_cont_features + sum([y for x, y in emb_dims])

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


class TablarNet_2(nn.Module):
    def __init__(self, emb_dims, cfg, in_cont_features=875, out_features=206):
        super(TablarNet_2, self).__init__()
        h = cfg.train.hidden_size
        d = cfg.train.dropout_rate

        self.embedding_layer = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        self.dropout = nn.Dropout(cfg.train.dropout_rate, inplace=True)

        self.first_bn_layer = nn.Sequential(
            nn.BatchNorm1d(in_cont_features),
            nn.Dropout(0.2)
        )

        first_in_feature = in_cont_features + sum([y for x, y in emb_dims])

        self.block = nn.Sequential(
            LinearReluBnDropout(in_features=first_in_feature, out_features=h, dropout_rate=d),
            LinearReluBnDropout(in_features=h, out_features=h // 2, dropout_rate=d),
            LinearReluBnDropout(in_features=h // 2, out_features=h // 4, dropout_rate=d)
        )

        self.last = nn.Linear(h // 4, out_features)

    def forward(self, cont_f, cat_f):

        cat_x = [layer(cat_f[:, i]) for i, layer in enumerate(self.embedding_layer)]
        cat_x = torch.cat(cat_x, 1)
        cat_x = self.dropout(cat_x)

        cont_x = self.first_bn_layer(cont_f)

        x = torch.cat([cont_x, cat_x], 1)

        x = self.block(x)
        x = self.last(x)

        return x


class TablarNet_res(nn.Module):
    def __init__(self, emb_dims, cfg, in_cont_features=875, out_features=206):
        super(TablarNet_res, self).__init__()

        self.embedding_layer = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        self.dropout = nn.Dropout(cfg.train.dropout_rate, inplace=True)

        self.first_bn_layer = nn.Sequential(
            nn.BatchNorm1d(in_cont_features),
            nn.Dropout(0.2)
        )

        first_in_feature = in_cont_features + sum([y for x, y in emb_dims])

        self.block = nn.Sequential(
            LinearReluBnDropout(in_features=first_in_feature,
                                out_features=cfg.train.hidden_size,
                                dropout_rate=cfg.train.dropout_rate),
            ResidualUnit(in_features=cfg.train.hidden_size,
                         out_features=cfg.train.hidden_size // 2,
                         dropout_rate=cfg.train.dropout_rate),
            ResidualUnit(in_features=cfg.train.hidden_size //  2,
                         out_features=cfg.train.hidden_size // 2,
                         dropout_rate=cfg.train.dropout_rate),
            ResidualUnit(in_features=cfg.train.hidden_size //  2,
                         out_features=cfg.train.hidden_size // 4,
                         dropout_rate=cfg.train.dropout_rate),
            ResidualUnit(in_features=cfg.train.hidden_size //  4,
                         out_features=cfg.train.hidden_size // 4,
                         dropout_rate=cfg.train.dropout_rate),
            ResidualUnit(in_features=cfg.train.hidden_size //  4,
                         out_features=cfg.train.hidden_size // 4,
                         dropout_rate=cfg.train.dropout_rate)
        )

        self.last = nn.Linear(cfg.train.hidden_size // 4, out_features)

    def forward(self, cont_f, cat_f):
        cat_x = [layer(cat_f[:, i]) for i, layer in enumerate(self.embedding_layer)]
        cat_x = torch.cat(cat_x, 1)
        cat_x = self.dropout(cat_x)

        cont_x = self.first_bn_layer(cont_f)

        x = torch.cat([cont_x, cat_x], 1)

        x = self.block(x)
        x = self.last(x)

        return x
