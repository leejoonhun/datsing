import pytorch_lightning as pl
from pytorch_forecasting.metrics import SMAPE
from pytorch_forecasting.models import NBeats
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy

from data import load_datasets
from utils import parse_args


def pretrain(
    forecast_period: int,
    lookback_mult: int,
    batch_size: int,
    num_epoch: int,
    seed: int,
    **kwargs,
):
    # configuration
    pl.seed_everything(seed)

    # data
    trainset, validset = load_datasets(
        data_name="m4",
        lookback_len=forecast_period * lookback_mult,
        forecast_len=forecast_period,
    )
    trainloader, validloader = (
        trainset.to_dataloader(
            train=True, batch_size=batch_size, num_workers=8, pin_memory=True
        ),
        validset.to_dataloader(
            train=False, batch_size=batch_size, num_workers=8, pin_memory=True
        ),
    )

    # model
    nbeats = NBeats.from_dataset(trainset, learning_rate=3e-2, loss=SMAPE())
    nbeats.save_hyperparameters(ignore=["loss", "logging_metrics"])

    # train
    trainer = Trainer(
        gradient_clip_val=0.1,
        max_epochs=num_epoch,
        accelerator="cuda",
        strategy=DDPStrategy(find_unused_parameters=False),
        auto_lr_find=True,
    )
    trainer.fit(nbeats, trainloader, validloader)


def main():
    args = parse_args()
    pretrain(**vars(args))


if __name__ == "__main__":
    main()