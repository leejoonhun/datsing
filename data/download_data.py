import os
from os import path as osp
from pathlib import Path

DATA_ROOT = Path(__file__).parent.resolve()


def main():
    if not osp.exists(DATA_ROOT / "m4" / "Monthly-train.csv"):
        print("Downloading dataset..")
        os.system(
            "wget https://raw.githubusercontent.com/M4Competition/M4-methods/master/Dataset/Train/Monthly-train.csv -O data/m4/"
        )


if __name__ == "__main__":
    main()
