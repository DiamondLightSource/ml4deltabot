import os

import numpy as np
from sklearn.metrics import r2_score

from . import config as cfg
from .data import build_sequences, load_file


def evaluate_r2(model, X, Y, scaler_x, y_mean, y_std, label="test"):
    N, W, C = X.shape
    X_scaled = scaler_x.transform(X.reshape(-1, C)).reshape(N, W, C).astype(np.float32)

    Y_pred_norm = model.predict(X_scaled, batch_size=cfg.BATCH_SIZE, verbose=0)
    Y_pred = Y_pred_norm * y_std + y_mean

    print(f"\nR² on {label} set:")
    for i, name in enumerate(["x_out", "y_out", "z_out"]):
        print(f"  {name}: {r2_score(Y[:, i], Y_pred[:, i]):.6f}")
    print(f"  overall: {r2_score(Y, Y_pred):.6f}")

    mae_nm = np.mean(np.abs(Y - Y_pred), axis=0)
    print(f"\nMAE on {label} set (nm):")
    for i, name in enumerate(["x_out", "y_out", "z_out"]):
        print(f"  {name}: {mae_nm[i]:.2f} nm")
    print(f"  overall: {mae_nm.mean():.2f} nm")

    return Y_pred


def evaluate_per_file(model, test_paths, scaler_x, y_mean, y_std):
    print("\nPer-file R² on test set:")
    for p in test_paths:
        X, Y = build_sequences(load_file(p), cfg.WINDOW_SIZE)
        N, W, C = X.shape
        X_scaled = scaler_x.transform(X.reshape(-1, C)).reshape(N, W, C).astype(np.float32)
        Y_pred_norm = model.predict(X_scaled, batch_size=cfg.BATCH_SIZE, verbose=0)
        Y_pred = Y_pred_norm * y_std + y_mean
        scores = [r2_score(Y[:, i], Y_pred[:, i]) for i in range(3)]
        print(f"  {os.path.basename(p):40s}  "
              f"x={scores[0]:.4f}  y={scores[1]:.4f}  z={scores[2]:.4f}")
