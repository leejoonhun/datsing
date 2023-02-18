from os import path as osp
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from torch.utils import data as dt

from .download_data import download

DATA_ROOT = Path(__file__).parent.resolve()


class TimeSeriesDataset(dt.Dataset):
    def __init__(
        self, ts_data: List[np.ndarray], lookback: int, horizon: int, step: int
    ):
        self.ts_data = ts_data
        self.lookback = lookback
        self.horizon = horizon

        last_id, n_dropped, self.ids = 0, 0, {}
        for i, ts in enumerate(self.ts_data):
            num_data = (ts.shape[-1] - self.lookback - self.horizon + step) // step

            if ts.shape[-1] < self.horizon:
                n_dropped += 1
                continue

            if ts.shape[-1] < self.lookback + self.horizon:
                num_data = 1

            for j in range(num_data):
                self.ids[last_id + j] = (i, j * step)
            last_id += num_data

        if n_dropped:
            print(f"Dropped {n_dropped}/{len(self.ts_data)} time series due to length")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        ts_id, lookback_id = self.ids[idx]
        ts = self.ts_data[ts_id]
        if ts.shape[-1] < self.lookback + self.horizon:
            X = ts[:, : -self.horizon]
            X = np.pad(
                X,
                pad_width=((0, 0), (self.lookback - X.shape[-1], 0)),
                mode="constant",
                constant_values=0,
            )
            y = ts[:, -self.horizon :]
        else:
            X = ts[:, lookback_id : lookback_id + self.lookback]
            y = ts[
                :,
                lookback_id
                + self.lookback : lookback_id
                + self.lookback
                + self.horizon,
            ]
        return X, y


def load_dataloaders(
    data_name: str, lookback: int, horizon: int, valid_ratio: float, batch_size: int
) -> dt.Dataset:
    if data_name.lower() == "m4":
        if not osp.exists(DATA_ROOT / "m4"):
            download()
        df = pd.read_csv(DATA_ROOT / "m4/Monthly-train.csv").iloc[:, 1:]
    elif data_name.lower() == "tour":
        df = (
            pd.read_csv(DATA_ROOT / "tour/monthly_in.csv", header=None)
            .transpose()
            .iloc[:, 1:]
            .astype(np.float64)
        )
    data = list(df.values)
    for i, ts in enumerate(data):
        data[i] = ts[~np.isnan(ts)][None, :-horizon]

    valid_len = int(len(data) * valid_ratio)
    data_train, data_valid = data[:-valid_len], data[-valid_len:]

    trainset = TimeSeriesDataset(data_train, lookback, horizon, step=1)
    validset = TimeSeriesDataset(data_valid, lookback, horizon, step=1)
    return (
        dt.DataLoader(trainset, batch_size=batch_size, shuffle=True),
        dt.DataLoader(validset, batch_size=batch_size, shuffle=False),
    )
