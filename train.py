import os
import glob
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from src.lightning import LightningSystem, DataModule
from src.model import TablarNet, TablarNet_2
from src.utils import seed_everything
from pytorch_lightning import Trainer
from comet_ml import Experiment

from pytorch_lightning.callbacks import ModelCheckpoint

import warnings
warnings.filterwarnings('ignore')

# Config  ###########################
# Input Data Directory
data_dir = './input'
# CV
cv = MultilabelStratifiedKFold(n_splits=4)


@hydra.main('config.yml')
def main(cfg: DictConfig):
    cur_dir = hydra.utils.get_original_cwd()
    os.chdir(cur_dir)
    # Random Seed
    seed_everything(cfg.train.seed)

    # Lightning Data Module  ####################################################
    datamodule = DataModule(data_dir, cfg, cv)
    datamodule.prepare_data()
    target_cols = datamodule.target_cols

    # Model  ####################################################################
    emb_dims = [(2, 15), (3, 20), (2, 15)]
    # Adjust input dim (original + composition dim - category features)
    in_features = 875 + cfg.train.g_comp + cfg.train.c_comp - 3
    net = TablarNet_2(emb_dims, cfg, in_cont_features=in_features)

    # Comet.ml
    experiment = Experiment(api_key=cfg.comet_ml.api_key,
                            project_name=cfg.comet_ml.project_name)
    # Log Parameters
    experiment.log_parameters(dict(cfg.exp))
    experiment.log_parameters(dict(cfg.train))
    # Log Model Graph
    experiment.set_model_graph(str(net))

    # Lightning Module  #########################################################
    model = LightningSystem(net, cfg, experiment, target_cols)

    checkpoint_callback = ModelCheckpoint(
        filepath='./checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='avg_val_loss',
        mode='min',
        prefix=cfg.exp.exp_name + '_'
    )

    trainer = Trainer(
        logger=False,
        max_epochs=cfg.train.epoch,
        checkpoint_callback=checkpoint_callback,
        # gpus=1
            )

    # Train & Test  ############################################################
    # Train
    trainer.fit(model, datamodule=datamodule)
    checkpoint_path = glob.glob(f'./checkpoint/{cfg.exp.exp_name}_*.ckpt')[0]
    experiment.log_asset(file_data=checkpoint_path)

    # Test
    # trainer.test(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
