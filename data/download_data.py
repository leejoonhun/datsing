import os
from os import path as osp


def download() -> None:
    os.system(
        "wget https://raw.githubusercontent.com/M4Competition/M4-methods/master/Dataset/Train/Monthly-train.csv -O data/m4/"
    )
