import pandas as pd
import itertools
import glob
import gc
from category_encoders import OrdinalEncoder
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os
from .dataset import MoADataset

from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.metrics.classification import AUROC


class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, cfg, cv):
        super(DataModule, self).__init__()
        self.cfg = cfg
        self.data_dir = data_dir
        self.cv = cv

    def prepare_data(self):
        # Prepare Data
        train_target = pd.read_csv(os.path.join(self.data_dir, 'train_targets_scored.csv'))
        train_feature = pd.read_csv(os.path.join(self.data_dir, 'train_features.csv'))
        test = pd.read_csv(os.path.join(self.data_dir, 'test_features.csv'))

        train = pd.merge(train_target, train_feature, on='sig_id')
        self.target_cols = [c for c in train_target.columns if c != 'sig_id']
        self.feature_cols = [c for c in train_feature.columns if c != 'sig_id']

        test['is_train'] = 0
        train['is_train'] = 1
        self.df = pd.concat([train, test], axis=0, ignore_index=True)
        self.test_id = test['sig_id'].values

        del train, train_target, train_feature, test
        gc.collect()

    def setup(self, stage=None):
        # Label Encoder
        cols = ['cp_type', 'cp_time', 'cp_dose']
        encoder = OrdinalEncoder(cols=cols)
        self.df = encoder.fit_transform(self.df)

        # Split Train, Test
        df = self.df[self.df['is_train'] == 1].reset_index(drop=True)
        test = self.df[self.df['is_train'] == 0].reset_index(drop=True)

        # Split Train, Validation
        df['fold'] = -1
        for i, (trn_idx, val_idx) in enumerate(self.cv.split(df)):
            df.loc[val_idx, 'fold'] = i
        fold = self.cfg.train.fold
        train = df[df['fold'] != fold].reset_index(drop=True)
        val = df[df['fold'] == fold].reset_index(drop=True)

        self.train_dataset = MoADataset(train, self.feature_cols, self.target_cols, phase='train')
        self.val_dataset = MoADataset(val, self.feature_cols, self.target_cols, phase='train')
        self.test_dataset = MoADataset(test, self.feature_cols, self.target_cols, phase='test')

        del encoder, df, test, train, val
        gc.collect()

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.cfg.train.batch_size,
                          pin_memory=True,
                          sampler=RandomSampler(self.train_dataset), drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.cfg.train.batch_size,
                          pin_memory=True,
                          sampler=SequentialSampler(self.val_dataset), drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.cfg.train.batch_size,
                          pin_memory=False,
                          shuffle=False, drop_last=False)


class LightningSystem(pl.LightningModule):
    def __init__(self, net, cfg, experiment):
        super(LightningSystem, self).__init__()
        self.net = net
        self.cfg = cfg
        self.experiment = experiment
        self.criterion = nn.BCEWithLogitsLoss()
        self.best_loss = 1e+9
        self.best_auc = None
        self.epoch_num = 0

    def configure_optimizers(self):
        self.optimizer = optim.AdamW(self.parameters(), lr=self.cfg.train.lr, weight_decay=2e-5)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.cfg.train.epoch, eta_min=0)

        return [self.optimizer], [self.scheduler]

    def forward(self, x):
        return self.net(x)

    def step(self, batch):
        inp, label = batch
        out = self.forward(inp)
        loss = self.criterion(out, label)

        return loss, label, torch.sigmoid(out)

    def training_step(self, batch, batch_idx):
        loss, label, logits = self.step(batch)

        logs = {'train/loss': loss.item()}
        # batch_idx + Epoch * Iteration
        step = batch_idx
        self.experiment.log_metrics(logs, step=step)

        return {'loss': loss, 'logits': logits, 'labels': label}

    def validation_step(self, batch, batch_idx):
        loss, label, logits = self.step(batch)

        val_logs = {'val/loss': loss.item()}
        # batch_idx + Epoch * Iteration
        step = batch_idx
        self.experiment.log_metrics(val_logs, step=step)

        return {'val_loss': loss, 'logits': logits.detach(), 'labels': label.detach()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'val/epoch_loss': avg_loss.item()}
        # Log loss
        self.experiment.log_metrics(logs, step=self.epoch_num)
        # Update Epoch Num
        self.epoch_num += 1

        # Save Weights
        if self.best_loss > avg_loss:
            self.best_loss = avg_loss
            filename = f'{self.cfg.exp.exp_name}_epoch_{self.epoch_num}_loss_{self.best_loss:.3f}.pth'
            torch.save(self.net.state_dict(), filename)
            self.experiment.log_model(name=filename, file_or_folder='./'+filename)
            os.remove(filename)

        return {'avg_val_loss': avg_loss}

    def test_step(self, batch, batch_idx):
        inp, img_name = batch
        out = self.forward(inp)
        logits = torch.sigmoid(out)

        return {'preds': logits, 'image_names': img_name}

    def test_epoch_end(self, outputs):
        PREDS = torch.cat([x['preds'] for x in outputs]).reshape((-1)).detach().cpu().numpy()
        # [tuple, tuple]
        IMG_NAMES = [x['image_names'] for x in outputs]
        # [list, list]
        IMG_NAMES = [list(x) for x in IMG_NAMES]
        IMG_NAMES = list(itertools.chain.from_iterable(IMG_NAMES))

        res = pd.DataFrame({
            'image_name': IMG_NAMES,
            'target': PREDS
        })
        try:
            res['target'] = res['target'].apply(lambda x: x.replace('[', '').replace(']', ''))
        except:
            pass
        N = len(glob.glob(f'submission_{self.cfg.exp.exp_name}*.csv'))
        filename = f'submission_{self.cfg.exp.exp_name}_{N}.csv'
        res.to_csv(filename, index=False)

        return {'res': res}