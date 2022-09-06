import argparse
import os

import wandb
from easydict import EasyDict as edict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data import GLUEDataModule, glue_input_template
from model import Model

# Due to the issue described in
# https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
# we need to disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument(
        "--dataset", type=str, required=True, choices=glue_input_template.keys()
    )
    parser.add_argument("--max_seq_len", type=int, default=128)
    # model
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    # training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--fp16", action="store_true")
    # experiment
    parser.add_argument("--project_id", type=str, default="bert-glue")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    config = edict(vars(parser.parse_args()))
    return config


def main():
    config = parse_args()
    # set seed
    seed_everything(config.seed)
    # setup data module, model, and trainer
    datamodule = GLUEDataModule(config)
    model = Model(config)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            filename="{epoch}-{avg_train_acc:.4f}-{avg_val_acc:.4f}",
            monitor="epoch",
            save_top_k=5,
            mode="max",
        ),
    ]
    logger = WandbLogger(
        offline=not config.wandb,
        project=config.project_id,
        entity="chrisliu298",
        config=config,
    )
    trainer = Trainer(
        accelerator="gpu",
        devices=-1,
        callbacks=callbacks,
        max_epochs=config.max_epochs,
        check_val_every_n_epoch=1,
        benchmark=True,
        logger=logger,
        enable_progress_bar=config.verbose > 0,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        precision=16 if config.fp16 else 32,
    )
    # run
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, verbose=config.verbose > 0)
    wandb.finish(quiet=config.verbose == 0)


if __name__ == "__main__":
    main()
