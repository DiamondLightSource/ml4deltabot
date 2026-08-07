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


_GROUP_BY_PREFIX = {
    "x_out": "position", "y_out": "position", "z_out": "position",
    "vx": "velocity", "vy": "velocity", "vz": "velocity",
    "x_in": "voltage", "y_in": "voltage", "z_in": "voltage",
}


def make_group_mae_metrics(y_std, target_names):
    """
    One MAE metric per target group (position/velocity), each in its
    own physical units.
    """
    y_std_t = tf.constant(y_std, dtype=tf.float32)

    groups = {}
    for i, name in enumerate(target_names):
        groups.setdefault(_GROUP_BY_PREFIX.get(name, "other"), []).append(i)

    metrics = []
    for group, idxs in groups.items():
        idxs_t = tf.constant(idxs, dtype=tf.int32)

        def metric_fn(y_true, y_pred, idxs_t=idxs_t):
            std = tf.gather(y_std_t, idxs_t)
            yt = tf.gather(tf.cast(y_true, tf.float32), idxs_t, axis=1) * std
            yp = tf.gather(tf.cast(y_pred, tf.float32), idxs_t, axis=1) * std
            return tf.reduce_mean(tf.abs(yt - yp))

        metric_fn.__name__ = f"physical_mae_{group}"
        metrics.append(metric_fn)

    return metrics
