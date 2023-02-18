from typing import Tuple

import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from torch.utils import data as dt

from .download_data import DATA_ROOT
from .download_data import main as download


def load_datasets(
    data_name: str, lookback_len: int, forecast_len: int
) -> Tuple[dt.DataLoader]:
    download()
    if data_name.lower() == "m4":
        df = (
            pd.read_csv(DATA_ROOT / "m4" / "Monthly-train.csv", index_col=0)
            .transpose()
            .reset_index()
            .drop("index", axis=1)
            .reset_index()
            .melt("index")
            .dropna()
            .rename(columns={"V1": "variable"})
        )
    elif data_name.lower() == "tour":
        df = (
            pd.read_csv(DATA_ROOT / "tour" / "monthly_in.csv", header=None)
            .reset_index()
            .melt("index")
            .dropna()
        )

    cutoff = df["index"].max() - forecast_len
    trainset = trainset = TimeSeriesDataSet(
        df[lambda x: x["index"] <= cutoff],
        time_idx="index",
        target="value",
        categorical_encoders={"variable": NaNLabelEncoder().fit(df["variable"])},
        group_ids=["variable"],
        time_varying_unknown_reals=["value"],
        min_encoder_length=lookback_len,
        max_encoder_length=lookback_len,
        max_prediction_length=forecast_len,
    )
    validset = TimeSeriesDataSet.from_dataset(
        trainset, df, min_prediction_idx=cutoff + 1
    )
    return trainset, validset
