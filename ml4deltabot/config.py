import os

DATA_DIR = "/dls/science/users/qfc83269/deltabot_data"

WINDOW_SIZE = 10
GRU_UNITS = 64
DENSE_UNITS = 64
BATCH_SIZE = 1024 * 4
EPOCHS = 500
LR = 1e-3
CLIPNORM = 1.0
TRAIN_RATIO = 0.8
SEED = 42
ES_PATIENCE = 40
LR_PATIENCE = 15
STEPS_PER_EXECUTION = 100

# Column indices in the raw txt file (10 columns total)
COL = dict(x_in=1, y_in=3, z_in=5, x_out=7, y_out=8, z_out=9)

MODEL_OUT = "gru_64.keras"
CHECKPOINT_OUT = "best_gru.keras"
NORM_STATS_OUT = "normalisation_stats_gru.npz"
LOSS_CURVE_OUT = "gru_loss_curve.png"
