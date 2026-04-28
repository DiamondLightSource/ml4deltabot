import glob
import os
import tensorflow as tf
import numpy as np
from typing import List, Tuple


# With window_size=W, feature count is:
#   inputs (x/y/z_in):  W*3
#   velocity:           (W-1)*3
#   acceleration:       (W-2)*3
#   jerk:               (W-3)*3
#   total:              (4W-6)*3
#
# window_size=4 = (16-6)*3 = 30 features
# window_size=5 = (20-6)*3 = 42 features

def compute_n_features(window_size: int) -> int:
    return (4 * window_size - 6) * 3


class DeltabotTFDataset:
    def __init__(
        self,
        filepaths: List[str],
        window_size: int = 4, # minimum 4 to include jerk
        batch_size: int = 256,
        shuffle_buffer: int = 10000,
    ):
        assert window_size >= 4, "window_size must be >= 4 to compute jerk"
        self.filepaths = filepaths
        self.window_size = window_size
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.n_features = compute_n_features(window_size)

    @staticmethod
    def _load_file(filepath):
        filepath = filepath.decode() if isinstance(filepath, bytes) else filepath
        if not isinstance(filepath, str):
            filepath = filepath.item().decode()

        data = np.fromfile(filepath, sep=" ", dtype=np.float64).reshape(-1, 10)

        x_in  = data[:, 1]
        y_in  = data[:, 3]
        z_in  = data[:, 5]
        x_out = data[:, 7]
        y_out = data[:, 8]
        z_out = data[:, 9]

        # Stack: [x_in, y_in, z_in, x_out, y_out, z_out]
        stacked = np.stack(
            [x_in, y_in, z_in, x_out, y_out, z_out], axis=1
        )

        return stacked.astype(np.float32)

    def _file_to_dataset(self, filepath):
        data = tf.numpy_function(self._load_file, [filepath], tf.float32)
        data.set_shape([None, 6])

        ds = tf.data.Dataset.from_tensor_slices(data)
 
        # Create sliding windows
        ds = ds.window(self.window_size, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(self.window_size))

        return ds.map(self._split_xy, num_parallel_calls=tf.data.AUTOTUNE)

    def _split_xy(self, window):
        # window shape: (W, 6)
        # Columns: [x_in, y_in, z_in, x_out, y_out, z_out]

        x_in  = window[:, 0:1]
        y_in  = window[:, 1:2]
        z_in  = window[:, 2:3]
        x_out = window[:, 3:4]
        y_out = window[:, 4:5]
        z_out = window[:, 5:6]

        Y = tf.concat([x_out[-1], y_out[-1], z_out[-1]], axis=0)

        pos = tf.concat([x_out, y_out, z_out], axis=1)  # (W, 3)

        # Discrete derivatives (Δt = 1)
        vel = pos[1:] - pos[:-1]
        acc = vel[1:] - vel[:-1]
        jerk = acc[1:] - acc[:-1]

        inputs_flat = tf.reshape(
            tf.concat([x_in, y_in, z_in], axis=1), [-1]
        )

        vel_flat  = tf.reshape(vel,  [-1])
        acc_flat  = tf.reshape(acc,  [-1])
        jerk_flat = tf.reshape(jerk, [-1])

        X = tf.concat([inputs_flat, vel_flat, acc_flat, jerk_flat], axis=0)

        X.set_shape([self.n_features])
        Y.set_shape([3])

        return X, Y

    def split_files(self, train_ratio=0.8):
        n = len(self.filepaths)
        split_idx = int(n * train_ratio)

        return (
            self.filepaths[:split_idx],
            self.filepaths[split_idx:]
        )

    def _build_from_files(self, file_list):
        files_ds = tf.data.Dataset.from_tensor_slices(file_list)

        ds = files_ds.interleave(
            lambda f: self._file_to_dataset(f),
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False
        )

        ds = ds.shuffle(self.shuffle_buffer)
        ds = ds.cache()

        return ds

    def finalize(self, train_ds, test_ds):
        """ add batching and prefetch to train and testing datasets """
        train_ds = (
            train_ds
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        test_ds = (
            test_ds
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        return train_ds, test_ds

    def get_datasets(self, train_ratio=0.8):
        """ use 80% for training and 20% for testing """
        train_files, test_files = self.split_files(train_ratio)

        train_ds = self._build_from_files(train_files)
        test_ds = self._build_from_files(test_files)

        train_ds, test_ds = self.finalize(train_ds, test_ds)
        return train_ds, test_ds

    def compute_normalisation_stats(self, train_ds, n_samples=50000):
        """
        Compute z-score normalisation stats from the training set only.

        Args:
            train_ds:  the batched training dataset
            n_samples: how many samples to use for stat estimation

        Returns:
            dict with x_mean, x_std, y_mean, y_std  (all np.float32 arrays)
        """
        X_sample, Y_sample = [], []
        collected = 0

        for x_batch, y_batch in train_ds:
            X_sample.append(x_batch.numpy())
            Y_sample.append(y_batch.numpy())
            collected += x_batch.shape[0]
            if collected >= n_samples:
                break

        X_sample = np.concatenate(X_sample, axis=0)
        Y_sample = np.concatenate(Y_sample, axis=0)

        x_mean = X_sample.mean(axis=0).astype(np.float32)
        x_std  = X_sample.std(axis=0).astype(np.float32)
        y_mean = Y_sample.mean(axis=0).astype(np.float32)
        y_std  = Y_sample.std(axis=0).astype(np.float32)

        # Guard against dead/constant channels
        x_std = np.where(x_std < 1e-8, 1.0, x_std).astype(np.float32)
        y_std = np.where(y_std < 1e-8, 1.0, y_std).astype(np.float32)

        self.norm_stats = {
            "x_mean": x_mean, "x_std": x_std,
            "y_mean": y_mean, "y_std": y_std,
        }
        return self.norm_stats

    def apply_normalisation(self, train_ds, test_ds):
        """
        Apply z-score normalisation to both datasets using stats from training set.
        compute_normalisation_stats() must be called first.
        """
        assert hasattr(self, "norm_stats"), \
            "Call compute_normalisation_stats(train_ds) before apply_normalisation()"

        x_mean = tf.constant(self.norm_stats["x_mean"])
        x_std  = tf.constant(self.norm_stats["x_std"])
        y_mean = tf.constant(self.norm_stats["y_mean"])
        y_std  = tf.constant(self.norm_stats["y_std"])

        def normalize(x, y):
            return (x - x_mean) / x_std, (y - y_mean) / y_std

        train_ds = train_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds  = test_ds.map(normalize,  num_parallel_calls=tf.data.AUTOTUNE)

        return train_ds, test_ds

    def save_normalisation_stats(self, path="normalisation_stats.npz"):
        """Save stats to disk for use at inference time."""
        assert hasattr(self, "norm_stats"), \
            "Call compute_normalisation_stats() first"
        np.savez(path, **self.norm_stats)
        print(f"Saved normalisation stats to {path}")



def find_files(data_dir: str) -> list[str]:
    """ locate all the files in deltabot data folder """
    pattern = os.path.join(data_dir, "*.txt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No .txt files found in {data_dir!r}")
    print(f"Found {len(paths)} files:")
    for p in paths:
        print(f"  {os.path.basename(p)}")
    return paths


def check_dataset_sanity(ds, n_batches=10, label="dataset"):
    all_x, all_y = [], []

    for x_batch, y_batch in ds.take(n_batches):
        x = x_batch.numpy()
        y = y_batch.numpy()
        all_x.append(x)
        all_y.append(y)

    X = np.concatenate(all_x, axis=0)
    Y = np.concatenate(all_y, axis=0)

    print(f"\n=== {label} ===")
    print(f"  Samples inspected : {len(X)}")
    print(f"  X shape           : {X.shape}  (expected: (N, 30))")
    print(f"  Y shape           : {Y.shape}  (expected: (N, 3))")

    # Shape assertions
    assert X.shape[1] == X.shape[1], "X feature count mismatch"
    assert Y.shape[1] == 3, f"Y should have 3 targets, got {Y.shape[1]}"

    # No NaN / Inf
    assert np.all(np.isfinite(X)), "NaN or Inf found in X!"
    assert np.all(np.isfinite(Y)), "NaN or Inf found in Y!"
    print(f"  No NaNs/Infs      : ✓")

    assert X.dtype == np.float32, f"Expected float32, got {X.dtype}"
    assert Y.dtype == np.float32, f"Expected float32, got {Y.dtype}"
    print(f"  dtype float32     : ✓")

    return X, Y


if __name__ == "__main__":
    paths = find_files("/dls/science/users/qfc83269/deltabot_data")

    datasets = DeltabotTFDataset(paths)
    train_ds, test_ds = datasets.get_datasets()

    X_train, Y_train = check_dataset_sanity(train_ds, n_batches=20, label="TRAIN (raw)")
    X_test,  Y_test  = check_dataset_sanity(test_ds,  n_batches=10, label="TEST  (raw)")

    # Compute stats from training set only, then normalise both
    stats = datasets.compute_normalisation_stats(train_ds, n_samples=50000)
    print("\nNormalisation stats:")
    print(f"  x_mean range: [{stats['x_mean'].min():.4f}, {stats['x_mean'].max():.4f}]")
    print(f"  x_std  range: [{stats['x_std'].min():.4f},  {stats['x_std'].max():.4f}]")
    print(f"  y_mean:       {stats['y_mean']}")
    print(f"  y_std:        {stats['y_std']}")

    train_ds, test_ds = datasets.apply_normalisation(train_ds, test_ds)
    datasets.save_normalisation_stats("normalisation_stats.npz")

    X_train_n, Y_train_n = check_dataset_sanity(train_ds, n_batches=20, label="TRAIN (normalised)")
    X_test_n,  Y_test_n  = check_dataset_sanity(test_ds,  n_batches=10, label="TEST  (normalised)")
