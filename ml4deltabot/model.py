import tensorflow as tf
from tensorflow import keras


def build_gru(window_size, n_channels, gru_units, dense_units):
    """Input(W,C) -> GRU -> Dense(relu) -> Dense(3, linear) position output."""
    inp = keras.Input(shape=(window_size, n_channels), name="raw_window")

    x = keras.layers.GRU(
        gru_units,
        activation="tanh",
        recurrent_activation="sigmoid",
        recurrent_dropout=0.0,   # must be 0 for CuDNN kernel
        return_sequences=False,
        name="gru_encoder",
    )(inp)

    x = keras.layers.Dense(dense_units, activation="relu", name="dense_proj")(x)

    # cast to float32 explicitly - required under mixed_float16 policy
    out = keras.layers.Dense(3, activation="linear", name="position_xyz",
                              dtype="float32")(x)

    return keras.Model(inp, out, name="DeltabotGRU")


def make_physical_mae_metric(y_std):
    """MAE in physical units (nm), rather than normalised units. y_std is (3,)."""
    y_std_t = tf.constant(y_std, dtype=tf.float32)

    def physical_mae_nm(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32) * y_std_t
        y_pred = tf.cast(y_pred, tf.float32) * y_std_t
        return tf.reduce_mean(tf.abs(y_true - y_pred))

    physical_mae_nm.__name__ = "physical_mae_nm"
    return physical_mae_nm
