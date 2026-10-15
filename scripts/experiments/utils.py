from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_history(history: Sequence[float], save_path: str, show: bool = True) -> None:
    plt.plot(list(range(len(history))), history)
    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"History per epoch ({save_path})")
    plt.savefig(f"{save_path}")

    if show:
        plt.show()
    plt.close()


def plot_loss_history(
    train_history: Sequence[float],
    val_history: Sequence[float],
    save_path: str,
    best_epoch: int | None = None,
    show: bool = False,
) -> None:
    """Plot train and validation losses on a logarithmic scale."""
    if len(train_history) != len(val_history):
        raise ValueError("train and validation histories must have equal length")
    if any(loss <= 0 for loss in (*train_history, *val_history)):
        raise ValueError("loss values must be positive for a logarithmic scale")

    epochs = list(range(1, len(train_history) + 1))
    fig, ax = plt.subplots()
    ax.plot(epochs, train_history, label="Train loss")
    ax.plot(epochs, val_history, label="Validation loss")
    if best_epoch is not None:
        ax.axvline(best_epoch, color="gray", linestyle="--", label="Best epoch")

    ax.set_yscale("log")
    ax.grid(which="both")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L1 loss")
    ax.set_title("Loss history")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)

    if show:
        plt.show()
    plt.close(fig)
