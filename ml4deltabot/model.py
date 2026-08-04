import tensorflow as tf
from tensorflow import keras


def build_gru(window_size, n_channels, gru_units, dense_units, n_outputs=3):
    """
    Input(W,C) -> [GRU]xN -> Dense(relu) -> Dense(n_outputs, linear).
    gru_units: int (single layer) or list[int] (stacked layers).
    n_outputs: size of the target vector (e.g. 9 for position+velocity+voltage).
    """
    if isinstance(gru_units, int):
        gru_units = [gru_units]

    inp = keras.Input(shape=(window_size, n_channels), name="raw_window")

    x = inp
    for i, units in enumerate(gru_units):
        x = keras.layers.GRU(
            units,
            activation="tanh",
            recurrent_activation="sigmoid",
            recurrent_dropout=0.0,   # must be 0 for CuDNN kernel
            return_sequences=(i < len(gru_units) - 1),
            name=f"gru_{i}",
        )(x)

    x = keras.layers.Dense(dense_units, activation="relu", name="dense_proj")(x)

    # cast to float32 explicitly - required under mixed_float16 policy
    out = keras.layers.Dense(n_outputs, activation="linear", name="targets",
                              dtype="float32")(x)

    return keras.Model(inp, out, name="DeltabotGRU")


def make_physical_mae_metric(y_std):
    """MAE in physical units, rather than normalised units. y_std is (n_outputs,)."""
    y_std_t = tf.constant(y_std, dtype=tf.float32)

    def physical_mae(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32) * y_std_t
        y_pred = tf.cast(y_pred, tf.float32) * y_std_t
        return tf.reduce_mean(tf.abs(y_true - y_pred))

    physical_mae.__name__ = "physical_mae"
    return physical_mae
