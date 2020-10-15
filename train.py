import os
import glob
import hydra
from omegaconf import DictConfig
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from pytorch_lightning import Trainer
from comet_ml import Experiment
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.lightning import LightningSystem, DataModule
from src.model import SimpleDenseNet
from src.utils import seed_everything, check_oof_score, Datapreprocessing

import warnings
warnings.filterwarnings('ignore')


@hydra.main('config.yml')
def main(cfg: DictConfig):
    cur_dir = hydra.utils.get_original_cwd()
    os.chdir(cur_dir)
    # Random Seed
    seed_everything(cfg.train.seed)

    # Config  ###########################
    # Input Data Directory
    data_dir = './input'
    NFOLDS = cfg.train.fold
    # CV
    cv = MultilabelStratifiedKFold(n_splits=NFOLDS)

    # Comet.ml
    experiment = Experiment(api_key=cfg.comet_ml.api_key,
                            project_name=cfg.comet_ml.project_name)
    # Log Parameters
    experiment.log_parameters(dict(cfg.exp))
    experiment.log_parameters(dict(cfg.train))


    # Data Loading & Preprocessing  ############################
    dataprep = Datapreprocessing(data_dir, cfg, cv)
    trainval, test = dataprep.run()
    target_cols = dataprep.target_cols
    feature_cols = dataprep.feature_cols

    for fold in range(NFOLDS):
        print(f'Fold: {fold}')
        # Lightning Data Module  ####################################################
        dm = DataModule(trainval, test, cfg, feature_cols, target_cols, fold)
        # dm.prepare_data()
        # target_cols = dm.target_cols
        # feature_cols = dm.feature_cols

        # Model  ####################################################################
        net = SimpleDenseNet(cfg, in_features=len(feature_cols))

        if fold == 0:
            # Log Model Graph
            experiment.set_model_graph(str(net))

        # Lightning Module  #########################################################
        model = LightningSystem(net, cfg, experiment, target_cols, fold)

        checkpoint_callback = ModelCheckpoint(
            filepath='./checkpoint',
            save_top_k=1,
            verbose=False,
            monitor='avg_val_loss',
            mode='min',
            prefix=f'{cfg.exp.exp_name}_{fold}_'
        )

        early_stop_callback = EarlyStopping(
            monitor='avg_val_loss',
            min_delta=0.00,
            patience=3,
            verbose=False,
            mode='min'
        )

        trainer = Trainer(
            logger=False,
            max_epochs=cfg.train.epoch,
            checkpoint_callback=checkpoint_callback,
            early_stop_callback=early_stop_callback,
            gpus=1,
                )

        # Train & Test  ############################################################
        print('MoA Prediction')
        print(f'Feature Num: {len(feature_cols)}')

        # Train
        trainer.fit(model, datamodule=dm)
        checkpoint_path = glob.glob(f'./checkpoint/{cfg.exp.exp_name}_{fold}_*.ckpt')[0]
        experiment.log_asset(file_data=checkpoint_path, copy_to_tmp=False)

    score = check_oof_score(cfg)
    experiment.log_metric('oof_score', score)


if __name__ == '__main__':
    main()
