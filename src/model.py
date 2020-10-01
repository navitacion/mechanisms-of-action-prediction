from torch import nn


class DenseModel(nn.Module):
    def __init__(self, cfg, in_features=875, out_features=206):
        super(DenseModel, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(in_features, cfg.train.hidden_size),
            nn.BatchNorm1d(cfg.train.hidden_size),
            nn.Dropout(cfg.train.dropout_rate),
            nn.PReLU(),
            nn.Linear(cfg.train.hidden_size, cfg.train.hidden_size),
            nn.BatchNorm1d(cfg.train.hidden_size),
            nn.Dropout(cfg.train.dropout_rate),
            nn.PReLU()
        )
        self.last = nn.Linear(cfg.train.hidden_size, out_features)

    def forward(self, x):
        x = self.block(x)
        x = self.last(x)

        return x

