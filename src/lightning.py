import os
import gc
import glob
import itertools
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

from .dataset import MoADataset


# class DataModule(pl.LightningDataModule):
#     def __init__(self, data_dir, cfg, cv, fold):
#         super(DataModule, self).__init__()
#         self.cfg = cfg
#         self.data_dir = data_dir
#         self.cv = cv
#         self.fold = fold
#
#
#     def _Encode(self, df):
#         cp_time_encoder = {
#             48: 1,
#             72: 2,
#             24: 0
#         }
#
#         cp_dose_encoder = {
#             'D1': 0,
#             'D2': 1
#         }
#
#         df['cp_time'] = df['cp_time'].map(cp_time_encoder)
#         df['cp_dose'] = df['cp_dose'].map(cp_dose_encoder)
#
#         for c in ['cp_time', 'cp_dose']:
#             df[c] = df[c].astype(int)
#
#         return df
#
#
#     def _get_dummies(self, df):
#         df = pd.get_dummies(df, columns=['cp_time','cp_dose'])
#         return df
#
#
#     def _add_PCA(self, df, cfg):
#         # g-features
#         g_cols = [c for c in df.columns if 'g-' in c]
#         temp = PCA(n_components=cfg.train.g_comp, random_state=42).fit_transform(df[g_cols])
#         temp = pd.DataFrame(temp, columns=[f'g-pca_{i}' for i in range(cfg.train.g_comp)])
#         df = pd.concat([df, temp], axis=1)
#
#         # c-features
#         c_cols = [c for c in df.columns if 'c-' in c]
#         temp = PCA(n_components=cfg.train.c_comp, random_state=42).fit_transform(df[c_cols])
#         temp = pd.DataFrame(temp, columns=[f'c-pca_{i}' for i in range(cfg.train.c_comp)])
#         df = pd.concat([df, temp], axis=1)
#
#         del temp
#
#         return df
#
#
#     def _scaler(self, df, feature_cols, type='standard'):
#         for c in feature_cols:
#             if c in ['cp_type', 'cp_time', 'cp_dose']:
#                 continue
#
#             if type == 'standard':
#                 scaler = StandardScaler()
#             elif type == 'robust':
#                 scaler = RobustScaler()
#             else:
#                 scaler = None
#             df[c] = scaler.fit_transform(df[c].values.reshape((-1, 1)))
#
#         return df
#
#
#     def _variancethreshold(self, df, threshold=0.5):
#         targets = df[self.target_cols]
#         var_thresh = VarianceThreshold(threshold=threshold)
#         cols = [c for c in df.columns if c not in self.target_cols + ['sig_id', 'is_train', 'cp_type', 'cp_time', 'cp_dose']]
#         temp = var_thresh.fit_transform(df[cols])
#
#         out = df[['sig_id', 'is_train', 'cp_type']]
#         temp = pd.DataFrame(temp)
#         out = pd.concat([out, temp, targets], axis=1)
#
#         return out
#
#
#     def prepare_data(self):
#         # Prepare Data
#         train_target = pd.read_csv(os.path.join(self.data_dir, 'train_targets_scored.csv'))
#         train_feature = pd.read_csv(os.path.join(self.data_dir, 'train_features.csv'))
#         test = pd.read_csv(os.path.join(self.data_dir, 'test_features.csv'))
#
#         train = pd.merge(train_target, train_feature, on='sig_id')
#         self.target_cols = [c for c in train_target.columns if c != 'sig_id']
#
#         test['is_train'] = 0
#         train['is_train'] = 1
#         self.df = pd.concat([train, test], axis=0, ignore_index=True)
#
#         # "ctl_vehicle" is not in scope prediction
#         self.df = self.df[self.df['cp_type'] != "ctl_vehicle"].reset_index(drop=True)
#
#         # Preprocessing
#         self.df = self._get_dummies(self.df)
#         self.df = self._add_PCA(self.df, self.cfg)
#         if self.cfg.train.threshold is not None:
#             self.df = self._variancethreshold(self.df, threshold=self.cfg.train.threshold)
#         self.feature_cols = [c for c in self.df.columns if c not in self.target_cols + ['sig_id', 'is_train', 'cp_type', 'cp_time', 'cp_dose']]
#         # self.df = self._scaler(self.df, self.feature_cols, type='standard')
#
#
#     def setup(self, stage=None):
#
#         if stage == 'fit':
#             trainval = self.df[self.df['is_train'] == 1].reset_index(drop=True)
#
#             # Split Train, Validation
#             trainval['fold'] = -1
#             for i, (trn_idx, val_idx) in enumerate(self.cv.split(trainval, trainval[self.target_cols])):
#                 trainval.loc[val_idx, 'fold'] = i
#
#             train = trainval[trainval['fold'] != self.fold].reset_index(drop=True)
#             val = trainval[trainval['fold'] == self.fold].reset_index(drop=True)
#
#             self.train_dataset = MoADataset(train[self.feature_cols].values,
#                                             train[self.target_cols].values,
#                                             None,
#                                             phase='train')
#
#             self.val_dataset = MoADataset(val[self.feature_cols].values,
#                                           val[self.target_cols].values,
#                                           val['sig_id'].values,
#                                           phase='val')
#
#         if stage == 'test':
#             test = self.df[self.df['is_train'] == 0].reset_index(drop=True)
#             self.test_dataset = MoADataset(test[self.feature_cols].values,
#                                            None,
#                                            test['sig_id'].values,
#                                            phase='test')
#
#
#     def train_dataloader(self):
#         return DataLoader(self.train_dataset,
#                           batch_size=self.cfg.train.batch_size,
#                           pin_memory=True,
#                           sampler=RandomSampler(self.train_dataset), drop_last=False)
#
#     def val_dataloader(self):
#         return DataLoader(self.val_dataset,
#                           batch_size=self.cfg.train.batch_size,
#                           pin_memory=True,
#                           sampler=SequentialSampler(self.val_dataset), drop_last=False)
#
#     def test_dataloader(self):
#         return DataLoader(self.test_dataset,
#                           batch_size=self.cfg.train.batch_size,
#                           pin_memory=False,
#                           shuffle=False, drop_last=False)


class DataModule(pl.LightningDataModule):
    def __init__(self, trainval, test, cfg, feature_cols, target_cols, fold):
        super(DataModule, self).__init__()
        self.trainval = trainval
        self.test = test
        self.cfg = cfg
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.fold = fold

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == 'fit':
            train = self.trainval[self.trainval['fold'] != self.fold].reset_index(drop=True)
            val = self.trainval[self.trainval['fold'] == self.fold].reset_index(drop=True)

            self.train_dataset = MoADataset(train[self.feature_cols].values,
                                            train[self.target_cols].values,
                                            None,
                                            phase='train')

            self.val_dataset = MoADataset(val[self.feature_cols].values,
                                          val[self.target_cols].values,
                                          val['sig_id'].values,
                                          phase='val')

        if stage == 'test':
            self.test_dataset = MoADataset(self.test[self.feature_cols].values,
                                           None,
                                           self.test['sig_id'].values,
                                           phase='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.cfg.train.batch_size,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.cfg.train.batch_size,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.cfg.train.batch_size,
                          shuffle=False, drop_last=False)


class LightningSystem(pl.LightningModule):
    def __init__(self, net, cfg, experiment, target_cols, fold):
        super(LightningSystem, self).__init__()
        self.net = net
        self.cfg = cfg
        self.experiment = experiment
        self.target_cols = target_cols
        self.fold = fold
        self.criterion = nn.BCEWithLogitsLoss()
        self.best_loss = 1e+9
        self.epoch_num = 0

    def configure_optimizers(self):
        self.optimizer = optim.AdamW(self.parameters(), lr=self.cfg.train.lr, weight_decay=2e-5)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.cfg.train.epoch, eta_min=0)

        return [self.optimizer], [self.scheduler]


    def training_step(self, batch, batch_idx):
        inputs, label = batch
        out = self.net(inputs)
        loss = self.criterion(out, label)

        return {'loss': loss, 'labels': label}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        logs = {f'train/epoch_loss_{self.fold}': avg_loss.item()}
        # Log loss
        self.experiment.log_metrics(logs, step=self.epoch_num)

    def validation_step(self, batch, batch_idx):
        inputs, label, ids = batch
        out = self.net(inputs)
        loss = self.criterion(out, label)
        logit = torch.sigmoid(out)

        return {'val_loss': loss, 'labels': label.detach(), 'id': ids, 'pred': logit}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {f'val/epoch_loss_{self.fold}': avg_loss.item()}
        # Log loss
        self.experiment.log_metrics(logs, step=self.epoch_num)

        # Save Weights
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            filename = f'{self.cfg.exp.exp_name}_fold_{self.fold}_epoch_{self.epoch_num}_loss_{self.best_loss:.5f}.pth'
            torch.save(self.net.state_dict(), filename)
            self.experiment.log_model(name=filename, file_or_folder='./'+filename)
            os.remove(filename)

            # oof
            oof_preds = torch.cat([x['pred'] for x in outputs]).detach().cpu().numpy()
            oof = pd.DataFrame(oof_preds, columns=self.target_cols)
            ids = [x['id'] for x in outputs]
            ids = [list(x) for x in ids]
            ids = list(itertools.chain.from_iterable(ids))

            oof.insert(0, 'sig_id', ids)
            oof_name = 'oof_' + self.cfg.exp.exp_name + f'_fold{self.fold}' + '.csv'
            oof.to_csv(os.path.join('./output', oof_name), index=False)

        # Update Epoch Num
        self.epoch_num += 1

        return {'avg_val_loss': avg_loss}

    def test_step(self, batch, batch_idx):
        inputs, ids = batch
        out = self.forward(inputs)
        logits = torch.sigmoid(out)

        return {'pred': logits, 'id': ids}


    def test_epoch_end(self, outputs):
        preds = torch.cat([x['pred'] for x in outputs]).detach().cpu().numpy()
        res = pd.DataFrame(preds, columns=self.target_cols)

        ids = [x['id'] for x in outputs]
        ids = [list(x) for x in ids]
        ids = list(itertools.chain.from_iterable(ids))

        res.insert(0, 'sig_id', ids)

        res.to_csv(os.path.join('./output', self.cfg.exp.exp_name + '.csv'), index=False)

        return {}