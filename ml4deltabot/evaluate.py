import os

import numpy as np
from sklearn.metrics import r2_score

from . import config as cfg
from .data import build_sequences, load_file

_GROUP_BY_PREFIX = {
    "x_out": "position", "y_out": "position", "z_out": "position",
    "vx": "velocity", "vy": "velocity", "vz": "velocity",
    "x_in": "voltage", "y_in": "voltage", "z_in": "voltage",
}


def _target_group(name):
    return _GROUP_BY_PREFIX.get(name, "unknown")


def _predict(model, X, scaler_x, y_mean, y_std):
    N, W, C = X.shape
    X_scaled = scaler_x.transform(X.reshape(-1, C)).reshape(N, W, C).astype(np.float32)
    Y_pred_norm = model.predict(X_scaled, batch_size=cfg.BATCH_SIZE, verbose=0)
    return Y_pred_norm * y_std + y_mean


def evaluate_predictions(Y, Y_pred, target_names, label="test"):
    """
    Per-target R², percentile absolute error, and % within tolerance
    (cfg.TOLERANCES, keyed by group: position/velocity/voltage).
    """
    abs_err = np.abs(Y - Y_pred)

    print(f"\nMetrics on {label} set:")
    for i, name in enumerate(target_names):
        group = _target_group(name)
        r2 = r2_score(Y[:, i], Y_pred[:, i])
        p50, p90, p95, p99 = np.percentile(abs_err[:, i], [50, 90, 95, 99])
        tol = cfg.TOLERANCES.get(group)
        pass_rate = (abs_err[:, i] <= tol).mean() * 100 if tol is not None else float("nan")

        print(f"  {name:6s} [{group:8s}]  R2={r2:8.5f}  "
              f"p50={p50:9.3f}  p90={p90:9.3f}  p95={p95:9.3f}  p99={p99:9.3f}  "
              f"within_tol({tol})={pass_rate:5.1f}%")

    print(f"  overall R2: {r2_score(Y, Y_pred):.6f}")
    return abs_err


def evaluate_r2(model, X, Y, scaler_x, y_mean, y_std, target_names, label="test"):
    Y_pred = _predict(model, X, scaler_x, y_mean, y_std)
    evaluate_predictions(Y, Y_pred, target_names, label=label)
    return Y_pred


def evaluate_per_file(model, test_paths, scaler_x, y_mean, y_std, target_names):
    print("\nPer-file R² on test set:")
    for p in test_paths:
        X, Y, _ = build_sequences(load_file(p), cfg.WINDOW_SIZE)
        Y_pred = _predict(model, X, scaler_x, y_mean, y_std)
        scores = [r2_score(Y[:, i], Y_pred[:, i]) for i in range(len(target_names))]
        row = "  ".join(f"{n}={s:.4f}" for n, s in zip(target_names, scores))
        print(f"  {os.path.basename(p):40s}  {row}")
