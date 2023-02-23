import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--forecast_period", type=int, default=18, help="forecasting period"
    )
    parser.add_argument(
        "--lookback_mult", type=float, default=3, help="lookback period multiple"
    )
    parser.add_argument("--batch_size", type=int, default=int(2e5))
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=718)
    args = parser.parse_args()
    return args
