"""
Data loader for multi-file deltabot inverse model training.

This module loads data from multiple files (different voltages) and creates
TensorFlow datasets for efficient training with proper scaling.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from typing import List, Tuple, Optional, Union
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler


def load_single_file(filepath: str, voltage_level: float) -> pd.DataFrame:
    """
    Load a single data file and add voltage level as metadata.

    Args:
        filepath: Path to the data file
        voltage_level: Voltage level for this file (e.g., 1.5, 3.0, 6.0)

    Returns:
        DataFrame with voltage_level column added
    """
    df = pd.read_csv(filepath, sep=r"\s+", header=None)
    df.columns = [
        "timestep",
        "x_in_reg", "x_in",
        "y_in_reg", "y_in",
        "z_in_reg", "z_in",
        "x_out", "y_out", "z_out"
    ]

    # Add voltage level as a column
    df["voltage_level"] = voltage_level

    return df


class DeltabotDatasetFromDataFrame:
    """Create TensorFlow datasets from a DataFrame with voltage_level column."""

    def __init__(self, df: pd.DataFrame, window_length: int = 3):
        """
        Initialize the dataset creator from DataFrame.

        Args:
            df: DataFrame with columns including voltage_level
            window_length: Length of sliding window for temporal features
        """
        self.df = df
        self.window_length = window_length

        # Scalers for features and targets
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

    def process_dataframe(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process the DataFrame to extract features and targets.

        Returns:
            Tuple of (X, y) where:
                X: Features (N, 4) - [x_out, y_out, z_out, voltage_level]
                y: Targets (N, 18) - [windows + velocities + accelerations]
        """
        print("\nProcessing DataFrame...")

        voltage_levels = self.df["voltage_level"].values

        # Extract input voltages
        x_in = self.df["x_in"].values.reshape(-1, 1)
        y_in = self.df["y_in"].values.reshape(-1, 1)
        z_in = self.df["z_in"].values.reshape(-1, 1)

        # Extract output positions
        x_out = self.df["x_out"].values.reshape(-1, 1)
        y_out = self.df["y_out"].values.reshape(-1, 1)
        z_out = self.df["z_out"].values.reshape(-1, 1)

        x_in.setflags(write=False)
        y_in.setflags(write=False)
        z_in.setflags(write=False)

        x_in_window = np.squeeze(sliding_window_view(x_in, self.window_length, 0))
        y_in_window = np.squeeze(sliding_window_view(y_in, self.window_length, 0))
        z_in_window = np.squeeze(sliding_window_view(z_in, self.window_length, 0))

        x_out_truncated = np.delete(x_out, slice(0, self.window_length - 1))
        y_out_truncated = np.delete(y_out, slice(0, self.window_length - 1))
        z_out_truncated = np.delete(z_out, slice(0, self.window_length - 1))
        voltage_levels_truncated = voltage_levels[self.window_length - 1:]

        all_data_in = np.concatenate([x_in_window, y_in_window, z_in_window], axis=1)

        # Calculate velocities (first derivative)
        x_velocities = x_in_window[:, 1:] - x_in_window[:, :-1]
        y_velocities = y_in_window[:, 1:] - y_in_window[:, :-1]
        z_velocities = z_in_window[:, 1:] - z_in_window[:, :-1]

        # Calculate accelerations (second derivative)
        x_accelerations = x_velocities[:, 1:] - x_velocities[:, :-1]
        y_accelerations = y_velocities[:, 1:] - y_velocities[:, :-1]
        z_accelerations = z_velocities[:, 1:] - z_velocities[:, :-1]

        # Concatenate: windows + velocities + accelerations (targets)
        y = np.concatenate([
            all_data_in,
            x_velocities, y_velocities, z_velocities,
            x_accelerations, y_accelerations, z_accelerations
        ], axis=1)

        # Stack output positions + voltage level (features)
        X = np.column_stack([
            x_out_truncated,
            y_out_truncated,
            z_out_truncated,
            voltage_levels_truncated
        ])

        print(f"  Processed: X={X.shape}, y={y.shape}")
        print(f"  Features: [x_pos, y_pos, z_pos, voltage_level]")
        print(f"  Targets:  [9 voltage windows, 6 velocities, 3 accelerations]")

        unique_voltages = np.unique(voltage_levels_truncated)
        print(f"\n  Voltage levels in dataset: {unique_voltages}")
        for v in unique_voltages:
            count = np.sum(voltage_levels_truncated == v)
            print(f"    {v}V: {count} samples ({count/len(X)*100:.1f}%)")

        return X, y

    def create_temporal_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_split: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create temporal train/test split (preserves time order).

        Args:
            X: Features
            y: Targets
            train_split: Fraction of data for training (default: 0.8)

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        split_idx = int(len(X) * train_split)

        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]

        print(f"\nTemporal split:")
        print(f"  Train: {X_train.shape[0]} samples ({train_split*100:.0f}%)")
        print(f"  Test:  {X_test.shape[0]} samples ({(1-train_split)*100:.0f}%)")

        return X_train, X_test, y_train, y_test

    def create_tf_datasets(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = 1024,
        shuffle_buffer: int = 10000
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Create optimized TensorFlow datasets.

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            batch_size: Batch size for training
            shuffle_buffer: Buffer size for shuffling training data

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        # Create training dataset
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = (train_ds
                    .shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
                    .batch(batch_size)
                    .prefetch(tf.data.AUTOTUNE))

        # Create test dataset (no shuffling for consistent evaluation)
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_ds = (test_ds
                   .batch(batch_size)
                   .prefetch(tf.data.AUTOTUNE))

        print(f"\nTensorFlow datasets created:")
        print(f"  Batch size: {batch_size}")
        print(f"  Train batches per epoch: {len(train_ds)}")
        print(f"  Test batches per epoch: {len(test_ds)}")

        return train_ds, test_ds

    def get_datasets(
        self,
        train_split: float = 0.8,
        batch_size: int = 1024,
        shuffle_buffer: int = 10000,
        scale_data: bool = True
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int, Optional[StandardScaler], Optional[StandardScaler]]:
        """
        Args:
            train_split: Fraction for training (default: 0.8)
            batch_size: Batch size (default: 1024)
            shuffle_buffer: Shuffle buffer size (default: 10000)
            scale_data: Whether to scale the data (default: True)

        Returns:
            Tuple of (train_dataset, test_dataset, input_dim, output_dim, X_scaler, y_scaler)
        """
        # Process DataFrame
        X, y = self.process_dataframe()

        # Scale the data
        if scale_data:
            print("\nScaling data...")
            X = self.X_scaler.fit_transform(X)
            y = self.y_scaler.fit_transform(y)
            print("  Features scaled (positions + voltage_level)")
            print("  Targets scaled (voltages + derivatives)")

        # Temporal split
        X_train, X_test, y_train, y_test = self.create_temporal_split(
            X, y, train_split=train_split
        )

        # Create TensorFlow datasets
        train_ds, test_ds = self.create_tf_datasets(
            X_train, y_train, X_test, y_test,
            batch_size=batch_size,
            shuffle_buffer=shuffle_buffer
        )

        input_dim = X.shape[1]
        output_dim = y.shape[1]

        print(f"\nModel dimensions:")
        print(f"  Input:  {input_dim}  (x_pos, y_pos, z_pos, voltage_level)")
        print(f"  Output: {output_dim} (voltage windows + derivatives)")

        # Return scalers if data was scaled
        X_scaler = self.X_scaler if scale_data else None
        y_scaler = self.y_scaler if scale_data else None

        if scale_data:
            print(f"\nScalers available for inverse transform")

        return train_ds, test_ds, input_dim, output_dim, X_scaler, y_scaler


def create_datasets_from_dataframe(
    df: pd.DataFrame,
    window_length: int = 3,
    train_split: float = 0.8,
    batch_size: int = 1024,
    shuffle_buffer: int = 10000,
    scale_data: bool = True
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int, Optional[StandardScaler], Optional[StandardScaler]]:
    """
    Args:
        df: DataFrame with voltage_level column
        window_length: Sliding window length (default: 3)
        train_split: Train/test split ratio (default: 0.8)
        batch_size: Batch size (default: 1024)
        shuffle_buffer: Shuffle buffer size (default: 10000)
        scale_data: Whether to scale features and targets (default: True)

    Returns:
        Tuple of (train_dataset, test_dataset, input_dim, output_dim, X_scaler, y_scaler)

    Example:
        # Load files
        df_6V = load_single_file("6V_all-axis_fast.txt", 6.0)
        df_3V = load_single_file("3V_all-axis_fast.txt", 3.0)
        df_1p5V = load_single_file("1_5V_all-axis_fast.txt", 1.5)

        # Concatenate
        df_all = pd.concat([df_6V, df_3V, df_1p5V], ignore_index=True)

        # Create datasets
        train_ds, test_ds, in_dim, out_dim, X_scaler, y_scaler = \
            create_datasets_from_dataframe(df_all)

        # Train
        model = build_nn_model(in_dim, out_dim)
        model.fit(train_ds, validation_data=test_ds, epochs=100)
    """
    dataset_creator = DeltabotDatasetFromDataFrame(df, window_length=window_length)
    return dataset_creator.get_datasets(train_split, batch_size, shuffle_buffer, scale_data)
