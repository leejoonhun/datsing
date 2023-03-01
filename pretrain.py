import pytorch_lightning as pl
from pytorch_forecasting import metrics
from pytorch_forecasting.models import NBeats
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torch import cuda

from data import load_datasets
from utils import parse_args


def pretrain(
    forecast_period: int,
    lookback_mult: int,
    batch_size: int,
    loss_type: str,
    num_epoch: int,
    seed: int,
    **kwargs,
):
    # configuration
    pl.seed_everything(seed)

    # data
    trainset, validset = load_datasets(
        mode="pretrain",
        lookback_len=forecast_period * lookback_mult,
        forecast_len=forecast_period,
    )
    trainloader, validloader = (
        trainset.to_dataloader(
            train=True,
            batch_size=batch_size,
            num_workers=4 * cuda.device_count(),
            pin_memory=True,
        ),
        validset.to_dataloader(
            train=False,
            batch_size=batch_size,
            num_workers=4 * cuda.device_count(),
            pin_memory=True,
        ),
    )

    # model
    loss = getattr(metrics, loss_type.upper())()
    nbeats = NBeats.from_dataset(trainset, learning_rate=3e-2, loss=loss)

    # train
    checkpoint_callback = ModelCheckpoint(
        "pretrained",
        "{epoch:02d}-{val_loss:.2f}",
        every_n_epochs=1,
    )
    trainer = Trainer(
        logger=False,
        callbacks=[checkpoint_callback],
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
