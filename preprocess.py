import glob
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.linear_model import LinearRegression


def stack_daily_prices(dir: str = "./data/"):
    # Assume files are from same year
    # TODO: Use close price for now, extend later

    files = glob.glob(dir+"close*.csv")
    files.sort()
    out = []

    for file in files:
        raw = pd.read_csv(dir+file, index_col="minutes")
        date = file[-8:-4]

        # FIXME: Set arbitrary year to the most recent leap year 2020; data has 02/29;
        raw.index = pd.to_datetime("2020"+date+raw.index, format="%Y%m%d%X")
        raw.index = raw.index.set_names("datetime")

        assert raw.shape == (330, 75), "csv has different shape"

        out.append(raw)

    return pd.concat(out)


def remove_zeros(data):
    i = 0
    while (data.iloc[i] == 0).any():
        i += 1
    data = data[i:]  # Discard first i rows if zero entry exists

    data = data.replace(to_replace=0, method="ffill")  # Forward fill zeros

    nonan = not data.isna().sum().sum()
    nonzero = (data != 0).all().all()

    assert nonan and nonzero, "Data has NaN or zero entry"

    return data


def calculate_log_return(dir: str = "./data/", save=False):
    data = stack_daily_prices(dir)
    data = remove_zeros(data)
    ret = data.apply(np.log).diff(1).dropna()  # Calculate log returns
    if save:
        save_dir = dir + "processed/"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        ret.to_csv(save_dir + "logret.csv")

    return ret


def calculate_resid_return(ret,
                           lookback=60,
                           lookback_cov=252,
                           oos_start_month=8,
                           factorList=[3, 5],
                           verbose=False,
                           save_dir="./data/processed"
                           ):
    """
    Calculate residual returns based on PCA approach (2.1)
    Ref: https://math.nyu.edu/~avellane/AvellanedaLeeStatArb20090616.pdf
    """
    T, N = ret.shape
    M = lookback_cov

    oos_start_idx = np.where(ret.index.month >= oos_start_month)[0][0]
    assert oos_start_idx > lookback_cov, "oos_start_month too small"

    R = ret.values

    resid_ret = np.zeros((T-oos_start_idx, N), dtype=np.float32)

    for num_factor in factorList:
        for idx in range(T-oos_start_idx):
            t = idx + oos_start_idx
            window = R[t-M:t].T  # (N x M)
            mean = np.mean(window, axis=1, keepdims=True)
            vol = np.std(window, axis=1, ddof=1, keepdims=True)
            if np.any(vol == 0):
                # FIXME: vol=0 creates nan in standardized returns
                continue

            Y = (window - mean) / vol
            Corr = (Y @ Y.T) / (M-1)
            eigval, eigvec = np.linalg.eig(Corr)

            # Sort in descending order
            order = np.flip(np.argsort(eigval))
            eigval = eigval[order]
            eigvec = eigvec[order]

            omega = eigvec[:num_factor].T / vol  # factor loadings
            F = window.T[-lookback:] @ omega  # factors

            regr = LinearRegression(fit_intercept=False, n_jobs=48).fit(
                F, window.T[-lookback:, :])

            beta = regr.coef_

            Phi = np.eye(N) - (beta @ omega.T)

            resid_ret[idx] = Phi @ R[t]

        dir = f"{save_dir}/ResidRet_LookBack_{lookback}_LookBackCov_{lookback_cov}_NumFactor_{num_factor}_InitOOSMonth_{oos_start_month}"
        if verbose:
            print("Calculating Resid returns under:\n", dir)
            print("Mean resid return: ", np.mean(resid_ret, axis=0))
        np.save(dir, resid_ret)


def calculate_cumsum(data, lookback):
    """
    Returns residual cumulative returns time series from given data

    Input
    data: np.ndarray (T x N)
    lookback: int

    Output
    windows: np.ndarray (T x N x lookback) 
    """
    signal_length = lookback
    T, N = data.shape
    cumsums = np.cumsum(data, axis=0)
    windows = np.zeros((T-lookback, N, signal_length), dtype=np.float32)

    for t in range(lookback, T):
        if t == lookback:
            windows[t-lookback, :, :] = cumsums[t-lookback:t, :].T
        else:
            windows[t-lookback, :, :] = cumsums[t-lookback:t, :].T - \
                cumsums[t-lookback-1, :].reshape(-1, 1)

    return windows


if __name__ == "__main__":
    # data = calculate_log_return("./data/")
    # data = calculate_resid_return(data, verbose=True)
    # data = np.load("./data/processed/" +
    #                "ResidRet_LookBack_60_LookBackCov_252_NumFactor_3_InitOOSMonth_8.npy")
    # preprocess_cumsum(data, 60)
    data = pd.read_csv("./data/processed/logret.csv", index_col=0, parse_dates=[0])
    data = calculate_resid_return(data, factorList=[1, 3, 5, 10, 15], verbose=True)
    pass
