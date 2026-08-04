import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

from . import config as cfg
from .data import load_all_files, split_files
from .model import build_gru, make_physical_mae_metric
from .evaluate import evaluate_r2, evaluate_per_file
from .plotting import plot_loss_curves


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--window-size", type=int, default=cfg.WINDOW_SIZE)
    p.add_argument("--gru-units", type=str, default=str(cfg.GRU_UNITS),
                   help="comma-separated for stacked layers, e.g. 128,64,32")
    p.add_argument("--dense-units", type=int, default=cfg.DENSE_UNITS)
    p.add_argument("--tag", type=str, default=None,
                   help="run identifier used in output filenames; auto-generated if omitted")
    return p.parse_args()


def set_seeds(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)


def enable_mixed_precision():
    policy = keras.mixed_precision.Policy("mixed_float16")
    keras.mixed_precision.set_global_policy(policy)
    print(f"Compute dtype : {policy.compute_dtype}")
    print(f"Variable dtype: {policy.variable_dtype}")


def make_datasets(X_train_s, Y_train_n, X_test_s, Y_test_n, batch_size, seed):
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train_s, Y_train_n))
        .shuffle(buffer_size=100_000, seed=seed)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((X_test_s, Y_test_n))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, test_ds


def main():
    args = parse_args()
    window_size = args.window_size
    gru_units = [int(u) for u in args.gru_units.split(",")]
    dense_units = args.dense_units
    tag = args.tag or f"w{window_size}_g{'-'.join(map(str, gru_units))}_d{dense_units}"

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    set_seeds(cfg.SEED)
    enable_mixed_precision()

    print(f"TF version : {tf.__version__}")
    print(f"GPUs       : {tf.config.list_physical_devices('GPU')}")
    print(f"Run tag    : {tag}")

    train_paths, test_paths = split_files(cfg.DATA_DIR, cfg.TRAIN_RATIO, cfg.SEED)
    print(f"\nTrain files ({len(train_paths)}):")
    for p in train_paths:
        print(f"  {p}")
    print(f"\nTest files ({len(test_paths)}):")
    for p in test_paths:
        print(f"  {p}")

    print("\nLoading train files ...")
    X_train, Y_train, target_names = load_all_files(train_paths, "train", window_size=window_size)
    print("\nLoading test files ...")
    X_test, Y_test, _ = load_all_files(test_paths, "test", window_size=window_size)

    print("\nFitting scaler ...")
    N_tr, W, C = X_train.shape
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train.reshape(-1, C)).reshape(N_tr, W, C).astype(np.float32)

    N_te = X_test.shape[0]
    X_test_s = scaler.transform(X_test.reshape(-1, C)).reshape(N_te, W, C).astype(np.float32)

    y_mean = Y_train.mean(axis=0).astype(np.float32)
    y_std = Y_train.std(axis=0).astype(np.float32)
    y_std = np.where(y_std < 1e-8, 1.0, y_std)

    Y_train_n = ((Y_train - y_mean) / y_std).astype(np.float32)
    Y_test_n = ((Y_test - y_mean) / y_std).astype(np.float32)

    print(f"y_mean: {y_mean}")
    print(f"y_std:  {y_std}")

    norm_stats_out = f"normalisation_stats_gru_{tag}.npz"
    np.savez(norm_stats_out, x_mean=scaler.mean_, x_std=scaler.scale_,
              y_mean=y_mean, y_std=y_std)

    train_ds, test_ds = make_datasets(X_train_s, Y_train_n, X_test_s, Y_test_n,
                                        cfg.BATCH_SIZE, cfg.SEED)

    model = build_gru(window_size, C, gru_units, dense_units, n_outputs=len(target_names))
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(cfg.LR, clipnorm=cfg.CLIPNORM),
        loss="mae",
        metrics=[make_physical_mae_metric(y_std)],
        steps_per_execution=cfg.STEPS_PER_EXECUTION,
    )

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=cfg.LR_PATIENCE,
            min_lr=1e-6, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg.ES_PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            f"best_gru_{tag}.keras", monitor="val_loss", save_best_only=True, verbose=0,
        ),
    ]

    print("\nTraining ...")
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=cfg.EPOCHS,
        callbacks=callbacks,
    )

    evaluate_r2(model, X_test, Y_test, scaler, y_mean, y_std, target_names, label="test")
    evaluate_per_file(model, test_paths, scaler, y_mean, y_std, target_names)
    plot_loss_curves(history, f"gru_loss_curve_{tag}.png")

    model_out = f"gru_{tag}.keras"
    model.save(model_out)
    print(f"Saved {model_out} and {norm_stats_out}")


if __name__ == "__main__":
    main()
