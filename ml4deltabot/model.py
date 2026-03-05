"""
Neural network model architectures for inverse modeling.
"""

import tensorflow as tf
from typing import Literal


def build_nn_model(
    input_dim: int,
    output_dim: int,
    architecture: Literal['standard', 'wide', 'deep'] = 'standard',
    learning_rate: float = 0.001
) -> tf.keras.Model:
    """
    Build a neural network model with specified architecture.

    Args:
        input_dim: Number of input features
        output_dim: Number of output values
        architecture: Model architecture type ('standard', 'wide', or 'deep')
        learning_rate: Learning rate for Adam optimizer

    Returns:
        Compiled Keras model
    """
    if architecture == 'standard':
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(output_dim)
        ])
    elif architecture == 'wide':
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(output_dim)
        ])
    elif architecture == 'deep':
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(output_dim)
        ])
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )

    return model


def get_callbacks(
    patience: int = 30,
    reduce_lr_patience: int = 15,
    checkpoint_path: str = None
):
    """
    Get training callbacks.

    Args:
        patience: Patience for early stopping
        reduce_lr_patience: Patience for learning rate reduction
        checkpoint_path: Path to save model checkpoints (optional)

    Returns:
        List of Keras callbacks
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-7,
            verbose=1
        )
    ]

    if checkpoint_path:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        )

    return callbacks
