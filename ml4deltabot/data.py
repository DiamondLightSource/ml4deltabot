import glob
import os

import numpy as np

from . import config as cfg


def find_files(data_dir):
    paths = sorted(glob.glob(os.path.join(data_dir, "*_all_axis_*.txt")))
    if not paths:
        raise FileNotFoundError(f"No .txt files in {data_dir!r}")
    return paths


def load_file(path):
    """Load one txt file -> (N, 6) float32 array: [x_in,y_in,z_in,x_out,y_out,z_out]."""
    data = np.fromfile(path, sep=" ", dtype=np.float64).reshape(-1, 10)
    return np.stack([
        data[:, cfg.COL["x_in"]],
        data[:, cfg.COL["y_in"]],
        data[:, cfg.COL["z_in"]],
        data[:, cfg.COL["x_out"]],
        data[:, cfg.COL["y_out"]],
        data[:, cfg.COL["z_out"]],
    ], axis=1).astype(np.float32)


def build_sequences(data, window_size):
    """
    Sliding windows: X (N-W, W, 6) raw sequence, Y (N-W, 3) position at next step
    (one-step-ahead prediction).
    """
    N, W = len(data), window_size
    n_seq = N - W
    idx = np.arange(W)[None, :] + np.arange(n_seq)[:, None]
    X = data[idx]
    Y = data[W:, 3:]
    return X.astype(np.float32), Y.astype(np.float32)


def load_all_files(paths, label):
    X_list, Y_list = [], []
    for p in paths:
        X, Y = build_sequences(load_file(p), cfg.WINDOW_SIZE)
        X_list.append(X)
        Y_list.append(Y)
        print(f"  {os.path.basename(p):40s}  {len(X):>8,} seqs")
    X_all = np.vstack(X_list)
    Y_all = np.vstack(Y_list)
    print(f"  {'Total ' + label:40s}  {len(X_all):>8,} seqs  "
          f"X={X_all.shape}  Y={Y_all.shape}")
    return X_all, Y_all


def split_files(data_dir, train_ratio, seed):
    paths = find_files(data_dir)
    rng = np.random.default_rng(seed)
    paths = list(rng.permutation(paths))
    split = int(len(paths) * train_ratio)
    return paths[:split], paths[split:]
