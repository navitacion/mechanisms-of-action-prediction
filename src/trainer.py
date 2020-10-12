import os, time, datetime
import numpy as np
import pandas as pd
import torch


class Trainer:
    def __init__(self, dataloaders, cfg, net, experiment, target_cols, device, criterion, optimizer, scheduler=None, checkpoint_path='./'):
        self.dataloaders = dataloaders
        self.cfg = cfg
        self.net = net
        self.experiment = experiment
        self.target_cols = target_cols
        self.device = device

        self.net = self.net.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_loss = 1e+9
        self.best_weight = None
        self.checkpoint_path = checkpoint_path


    def train(self, epoch=None):
        self.net.train()
        phase = 'train'
        epoch_loss = 0

        for input, target in self.dataloaders[phase]:
            self.optimizer.zero_grad()

            input = input.to(self.device)
            target = target.to(self.device)

            out = self.net(input)
            loss = self.criterion(out, target)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(self.dataloaders[phase])
        logs = {'train/epoch_loss': epoch_loss}
        self.experiment.log_metrics(logs, step=epoch)

        print(f'Phase: Train  Epoch Loss: {epoch_loss:.5f}')

    def valid(self, epoch):
        self.net.eval()
        phase = 'val'
        epoch_loss = 0
        for input, target, _ in self.dataloaders[phase]:
            with torch.no_grad():
                input = input.to(self.device)
                target = target.to(self.device)

                out = self.net(input)
                loss = self.criterion(out, target)

                epoch_loss += loss.item()

        epoch_loss /= len(self.dataloaders[phase])
        logs = {'val/epoch_loss': epoch_loss}
        self.experiment.log_metrics(logs, step=epoch)

        if epoch_loss < self.best_loss:
            self.best_weight = self.net.load_state_dict
            weight_name = f"{self.cfg.exp.exp_name}_loss_{epoch_loss:.5f}_epoch_{epoch}.pth"
            torch.save(self.best_weight, os.path.join(self.checkpoint_path, weight_name))
            self.best_loss = epoch_loss
            self.experiment.log_asset(file_data=os.path.join(self.checkpoint_path, weight_name), copy_to_tmp=False)
            os.remove(os.path.join(self.checkpoint_path, weight_name))

        print(f'Phase: Val  Epoch Loss: {epoch_loss:.5f}')


    def oof(self):
        self.net.eval()
        phase = 'val'
        ids = []
        preds = []

        for input, _, sig_id in self.dataloaders[phase]:
            with torch.no_grad():
                input = input.to(self.device)

                out = self.net(input)
                out = torch.sigmoid(out)
                preds.append(out)
                ids.extend(sig_id)

        preds = torch.cat(preds).detach().cpu().numpy()
        oof = pd.DataFrame(preds, columns=self.target_cols)
        oof.insert(0, 'sig_id', ids)

        filename = f'oof_{self.cfg.exp.exp_name}_fold_{1}.csv'
        oof.to_csv(filename)
        self.experiment.log_asset(file_data=filename, copy_to_tmp=False)
        os.remove(filename)


    def test(self):
        pass


    def fit(self):
        for epoch in range(self.cfg.train.epoch):
            print('#'*30)
            print(f'Epoch {epoch}')
            self.train(epoch=epoch)
            self.valid(epoch=epoch)
            if self.scheduler is not None:
                self.scheduler.step()

        self.net.load_state_dict(self.best_weight)
        self.oof()
        self.test()


