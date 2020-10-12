import os
import glob
import hydra
from comet_ml import Experiment
from omegaconf import DictConfig
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.trainer import Trainer
from src.model import SimpleDenseNet, Model
from src.utils import seed_everything, Datapreprocessing, get_dataloaders

import warnings
warnings.filterwarnings('ignore')

# Config  ###########################
# Input Data Directory
data_dir = './input'
# CV
cv = MultilabelStratifiedKFold(n_splits=4)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

@hydra.main('config.yml')
def main(cfg: DictConfig):
    cur_dir = hydra.utils.get_original_cwd()
    os.chdir(cur_dir)
    # Random Seed
    seed_everything(cfg.train.seed)

    # Comet.ml
    experiment = Experiment(api_key=cfg.comet_ml.api_key,
                            project_name=cfg.comet_ml.project_name)
    # Log Parameters
    experiment.log_parameters(dict(cfg.exp))
    experiment.log_parameters(dict(cfg.train))

    print('MoA Prediction')
    print(f'Using Device: {device}')

    # Data Module  ####################################################
    dataprep = Datapreprocessing(data_dir, cfg, cv)
    trainval, test, feature_cols, target_cols = dataprep.run()

    fold = 1
    dataloaders = get_dataloaders(trainval, test, cfg.train.batch_size, fold, feature_cols, target_cols)

    # Model  ####################################################################
    # net = SimpleDenseNet(cfg, in_features=len(feature_cols))
    net = Model(num_features=len(feature_cols), num_targets=len(target_cols), hidden_size=1024)
    # Log Model Graph
    experiment.set_model_graph(str(net))

    # Trainer  #########################################################
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(net.parameters(), lr=cfg.train.lr, weight_decay=2e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.epoch, eta_min=0)
    trainer = Trainer(dataloaders, cfg, net, experiment, target_cols, device,
                      criterion, optimizer, scheduler)

    # Train & Test  ############################################################
    print('Start to Train')
    trainer.fit()


if __name__ == '__main__':
    main()
