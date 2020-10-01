import os
import glob
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import KFold

from src.lightning import LightningSystem, DataModule
from src.model import DenseModel
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
cv = KFold(n_splits=4, shuffle=True)


@hydra.main('config.yml')
def main(cfg: DictConfig):
    cur_dir = hydra.utils.get_original_cwd()
    os.chdir(cur_dir)
    # Random Seed
    seed_everything(cfg.train.seed)

    # Lightning Data Module  ####################################################
    datamodule = DataModule(data_dir, cfg, cv)

    # Model  ####################################################################
    net = DenseModel(cfg,
                     in_features=len(datamodule.feature_cols),
                     out_features=len(datamodule.target_cols))

    # Comet.ml
    experiment = Experiment(api_key=cfg.comet_ml.api_key,
                            project_name=cfg.comet_ml.project_name)
    # Log Parameters
    experiment.log_parameters(dict(cfg.exp))
    experiment.log_parameters(dict(cfg.train))
    # Log Model Graph
    experiment.set_model_graph(str(net))

    # Lightning Module  #########################################################
    model = LightningSystem(net, cfg, experiment)

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
    experiment.log_metric('best_auc', model.best_auc)
    checkpoint_path = glob.glob(f'./checkpoint/{cfg.exp.exp_name}_*.ckpt')[0]
    experiment.log_asset(file_data=checkpoint_path)

    # Test
    # for i in range(test_num):
    #     trainer.test(model)


if __name__ == '__main__':
    main()
