import os
import time
import glob
import torch
import numpy as np
import pandas as pd

from datetime import datetime
from model import CNNTransformer
from pathlib import Path

from train import train, validate
from preprocess import *


def run(model, data, **kwargs):
    start = time.time()

    T, N = data.shape
    split = int(T * 0.7)  # Use 70% as training set

    data_train = data[:split]
    data_test = data[split:]

    print("Start training\n")
    train(model, data_train, log_freq=1, save_params=True, **kwargs)

    end = time.time()

    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(
        f"Total training time: {time.strftime('%H:%M:%S', time.gmtime(end-start))}")

    print("Start validation")
    result = validate(model, data_test, **kwargs)

    # for out in result:
    #     print(out)


def main():
    # TODO: add argparse handler
    seed = 12
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    print("Time: ", now)
    print(f"Seed: {seed}")

    preset = f"./logs/{now}/"

    Path(preset).mkdir(parents=True, exist_ok=True)
    print(f"\nPreset: {preset}\n")

    data_dir = "./data/"
    print(f"Loading data from {data_dir}")

    filepaths = glob.glob(data_dir + "processed/*.npy")
    if not filepaths:
        print("No preprocessed data found. Start preprocessing")
        raw_filepaths = glob.glob(data_dir + "logret.csv")
        if not raw_filepaths:
            stack_daily_prices(data_dir)
            calculate_log_return(data_dir, save=True)

        data = pd.read_csv(data_dir + "processed/logret.csv",
                           index_col=0, parse_dates=[0])
        data = calculate_resid_return(
            data, factorList=[1, 3, 5, 10, 15], verbose=True)

        filepaths = glob.glob(data_dir + "processed/*.npy")

    for path in filepaths:
        path = Path(path)
        tc = 0.0050
        # tc = float(input("tran_cost(bp): ")) * 0.0001
        print("trans_cost for this run: ", tc)
        tag = f"tc-{int(tc * 10000)}bp/"
        logdir = os.path.join(preset, tag, f"{path.name[:-4]}")
        Path(logdir).mkdir(parents=True, exist_ok=True)
        print(f"\nLogdir: {logdir}\n")

        data = np.load(path)

        print("Loading model")
        model = CNNTransformer(logdir)

        run(model, data, model_tag=tag, logdir=logdir, trans_cost=tc, seed=seed)


if __name__ == "__main__":
    main()
