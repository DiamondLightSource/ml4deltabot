import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_loss_curves(history, out_path):
    epochs_ran = range(1, len(history.history["loss"]) + 1)

    group_keys = sorted(
        k[len("physical_mae_"):] for k in history.history
        if k.startswith("physical_mae_")
    )

    fig, axes = plt.subplots(1 + len(group_keys), 1, figsize=(10, 4 * (1 + len(group_keys))), sharex=True)
    if len(group_keys) == 0:
        axes = [axes]

    ax0 = axes[0]
    ax0.plot(epochs_ran, history.history["loss"], label="train_loss (normalised)")
    ax0.plot(epochs_ran, history.history["val_loss"], label="val_loss (normalised)")
    ax0.set_ylabel("MAE (normalised)")
    ax0.set_title("GRU — Training vs Validation Loss")
    ax0.legend()
    ax0.grid(True)

    for ax, group in zip(axes[1:], group_keys):
        key = f"physical_mae_{group}"
        ax.plot(epochs_ran, history.history[key], label=f"train {group} MAE")
        ax.plot(epochs_ran, history.history[f"val_{key}"], label=f"val {group} MAE")
        ax.set_ylabel(f"{group} MAE (physical units)")
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")
