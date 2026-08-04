import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_loss_curves(history, out_path):
    epochs_ran = range(1, len(history.history["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(epochs_ran, history.history["loss"], label="train_loss (normalised)")
    ax1.plot(epochs_ran, history.history["val_loss"], label="val_loss (normalised)")
    ax1.set_ylabel("MAE (normalised)")
    ax1.set_title("GRU — Training vs Validation Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_ran, history.history["physical_mae"], label="train MAE")
    ax2.plot(epochs_ran, history.history["val_physical_mae"], label="val MAE")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MAE (physical units)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")
