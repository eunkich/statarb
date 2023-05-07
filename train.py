import os
import time

import torch
import numpy as np
import torch.nn as nn

from torch.optim import Adam
from preprocess import calculate_cumsum
from model import CNNTransformer
from datetime import datetime
from pathlib import Path


def train(model, data_train,
          num_epochs=100, batchsize=200, optim_kwargs={"lr": 0.001},
          save_params=True, logdir=None, model_tag='',
          lookback=60,
          trans_cost=0, hold_cost=0,
          parallelize=True, device=None, device_ids=[0],
          force_retrain=True,
          seed=12,
          verbose=True, log_freq=1,
          ):

    if logdir is None:
        logdir = model.logdir
    if device is None:
        device = model.device

    T, N = data_train.shape
    windows = calculate_cumsum(data_train, lookback)

    if verbose:
        print(f"train(): data_train.shape {data_train.shape}")
        print(f"train(): T {T} N {N}")
        print(f"train(): windows.shape {windows.shape}")

    if parallelize:
        model = nn.DataParallel(model, device_ids=device_ids)

    model.to(device)
    model.train()
    optimizer = Adam(model.parameters(), **optim_kwargs)

    # Handle Checkpoints
    already_trained = False
    # FIXME: handle filename
    checkpoint_fname = f'Checkpoint-seed_{seed}-{model_tag[:-1]}.tar'
    if os.path.isfile(os.path.join(logdir, checkpoint_fname)) and not force_retrain:
        already_trained = True
        checkpoint = torch.load(os.path.join(
            logdir, checkpoint_fname), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.train()
        print('Already trained!')

    begin_time = time.time()

    # Main Training loop
    for epoch in range(num_epochs):
        rets_train = np.zeros(T-lookback)
        short_proportion = np.zeros(T-lookback)
        turnover = np.zeros(T-lookback)
        # Assume there's no initial position
        last_weight = torch.zeros((1, N), device=device)

        # break input data up into batches of size `batchsize` and train over them
        num_batch = int((T-lookback)/batchsize)

        for i in range(num_batch + 1):
            # Handle index
            start = batchsize * i
            end = min(batchsize * (i + 1), T - lookback)

            weights = torch.zeros((end-start, N), device=device)
            dummy_idx = torch.ones_like(weights, dtype=bool, device="cpu")

            input_batch_i = windows[start:end][dummy_idx]

            weights[dummy_idx] = model(
                torch.tensor(input_batch_i, device=device))

            abs_sum = torch.sum(torch.abs(weights), axis=1, keepdim=True)
            weights = weights / abs_sum

            ret_res = torch.tensor(
                data_train[start+lookback:end+lookback, :], device=device)
            rets_batch = torch.sum(weights * ret_res, axis=1)

            short_proportion_batch = torch.sum(
                torch.abs(torch.min(weights, torch.zeros(1, device=device))), axis=1)
            turnover_batch = torch.diff(
                torch.cat((last_weight, weights)), axis=0)
            turnover_batch = torch.sum(torch.abs(turnover_batch), axis=1)
            last_weight[:] = weights[-1].detach().cpu()

            # TODO: Handle overnight positions to apply hold cost
            # - hold_cost * short_proportion_batch
            rets_batch = rets_batch - trans_cost * turnover_batch

            mean_ret = torch.mean(rets_batch)
            std = torch.std(rets_batch)

            loss = -mean_ret/std  # Maximize Sharpe ratio

            if not already_trained:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            weights = weights.detach().cpu().numpy()
            rets_train[start:end] = rets_batch.detach().cpu().numpy()
            turnover[start:end] = turnover_batch.detach().cpu().numpy()
            short_proportion[start:end] = short_proportion_batch.detach(
            ).cpu().numpy()

            full_ret = np.mean(rets_train)
            full_std = np.std(rets_train)
            full_sharpe = full_ret/full_std
            full_turnover = np.mean(turnover)
            full_short_proportion = np.mean(short_proportion)

        if verbose and epoch % log_freq == 0:
            # Train logging
            print(
                f"|Epoch: {epoch:3d}/{num_epochs} " +
                f"|Sharpe: {full_sharpe*np.sqrt(60 * 24 * 252):4.2f} " +
                f"|ret: {full_ret * 60 * 24 * 252:4.4f} " +
                f"|std: {full_std * np.sqrt(60 * 24 * 252):4.4f} " +
                f"|turnover: {full_turnover:0.3f} " +
                f"|short proportion: {full_short_proportion:0.3f} " +
                f"|time per epoch: {(time.time()-begin_time)/(epoch+1):0.2f}s |"
            )

            # TODO: Implement early stopping

        if already_trained:
            break

    if save_params and not already_trained:
        # can also save model.state_dict() directly w/o the dictionary; extension should then be .pth instead of .tar
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }
        checkpoint_fname = f'Checkpoint-seed_{seed}_{model_tag[:-1]}.tar'
        torch.save(checkpoint, os.path.join(logdir, checkpoint_fname))

    print(
        f'Training done - Model: {model_tag}, seed: {seed}')

    return rets_train, turnover, short_proportion, weights


def validate(model, data_test,
             lookback=60,
             device="cpu",
             trans_cost=0, hold_cost=0,
             save_result=True, logdir="./logs/", model_tag='',
             verbose=True,
             **kwargs
             ):
    # Validate performance on test dataset
    if device is None:
        device = model.device
    model.to(device)

    # Covert residual returns to cumulative returns
    T, N = data_test.shape
    windows = calculate_cumsum(data_test, lookback)

    rets_test = torch.zeros(T-lookback)
    short_proportion = np.zeros(T-lookback)
    turnover = np.zeros(T-lookback)
    weights = torch.zeros(T-lookback, N, device=device)

    dummy_idx = torch.ones_like(weights, dtype=bool, device="cpu")

    # break input data up into batches of size `batchsize` and train over them
    with torch.no_grad():
        # TODO: Extend to ensemble strategy with multiple checkpoints
        weights[dummy_idx] += model(torch.tensor(
            windows[dummy_idx], device=device))
        abs_sum = torch.sum(torch.abs(weights), axis=1, keepdim=True)
        weights = weights / abs_sum

        rets_test += torch.sum(weights*torch.tensor(
            data_test[lookback: T, :], device=device), axis=1)

        turnover = torch.cat((torch.zeros(1, device=device), torch.sum(
            torch.abs(weights[1:]-weights[:-1]), axis=1)))
        turnover[0] = torch.mean(turnover[1:])

        short_proportion = torch.sum(
            torch.abs(torch.min(weights, torch.zeros(1, device=device))), axis=1)

        weights = weights.detach().cpu().numpy()
        rets_test = rets_test - trans_cost * turnover  # - hold_cost * short_proportion
        rets_test = rets_test.detach().cpu().numpy()
        turnover = turnover.detach().cpu().numpy()
        short_proportion = short_proportion.detach().cpu().numpy()

        mean_ret = np.mean(rets_test)
        std = np.std(rets_test)
        sharpe = mean_ret/std  # Test Sharpe ratio
        mean_turnover = np.mean(turnover)
        mean_short_proportion = np.mean(short_proportion)
        if verbose:
            # Validation logging
            print(
                f"|Validation result|\n" + "-" * 20 + "\n"
                f"|Sharpe: {sharpe * np.sqrt(60 * 24 * 252):4.2f} " +
                f"|ret: {mean_ret * 60 * 24 * 252:4.4f} " +
                f"|std: {std * np.sqrt(60 * 24 * 252):0.4f} " +
                f"|turnover: {mean_turnover:0.3f} " +
                f"|short proportion: {mean_short_proportion:0.3f} "
            )

    print(f'Validation done - Model: {model_tag[:-1]}')
    if save_result:
        import pandas as pd
        rets_test *= 60 * 24 * 252  # Convert to Annual return
        results = np.vstack([rets_test, turnover, short_proportion]).T
        results = np.hstack([results, weights])
        df = pd.DataFrame(results, columns=[
                          "return", "turnover", "short_proportion"] + [f"weight_{i}" for i in range(75)])
        filename = f"ValidationResult.csv"
        df.to_csv(os.path.join(logdir, filename))

    return rets_test, turnover, short_proportion, weights


if __name__ == "__main__":
    seed = 12
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    print("Current Time: ", now)
    print(f"Seed: {seed}")
    start = time.time()

    logdir = f"./logs/"
    Path(logdir).mkdir(parents=True, exist_ok=True)
    print(f"\nlog dir: {logdir}\n")

    print("Loading model")
    model = CNNTransformer(logdir)

    print("Loading data")
    data = np.load("./data/processed/" +
                   "ResidRet_LookBack_60_LookBackCov_252_NumFactor_3_InitOOSMonth_8.npy")

    T, N = data.shape
    split = int(T * 0.7)  # Use 70% as training set

    data_train = data[:split]
    data_test = data[split:]

    print("Start training\n")
    train(model, data_train, log_freq=1, save_params=True, seed=seed)

    end = time.time()

    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(
        f"Total training time: {time.strftime('%H:%M:%S', time.gmtime(end-start))}")

    print("Start validation")
    result = validate(model, data_test)
    # for out in result:
    #     print(result)
