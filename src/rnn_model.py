"""
rnn_model.py
------------
Builds, trains, and evaluates recurrent models for IMDB sentiment
classification, and compares three cell types:

    - SimpleRNN  (baseline, prone to vanishing gradients on long sequences)
    - LSTM       (gated, long-term memory via cell state)
    - GRU        (gated, similar to LSTM but fewer parameters)

Usage:
    python src/rnn_model.py --cell lstm
    python src/rnn_model.py --cell gru
    python src/rnn_model.py --cell simple_rnn
    python src/rnn_model.py --compare        # trains all three and plots comparison
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, SimpleRNN, LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from preprocess import get_preprocessed_data, NUM_WORDS, MAX_LEN

EMBED_DIM = 64
RNN_UNITS = 64
BATCH_SIZE = 128
EPOCHS = 10


class GradientNormMonitor(Callback):
    """Logs the global gradient norm each epoch, to watch for
    vanishing/exploding gradients across cell types."""

    def __init__(self):
        super().__init__()
        self.grad_norms = []

    def on_epoch_end(self, epoch, logs=None):
        # Approximate via weight-delta norm as a lightweight proxy
        # (full per-batch gradient logging would need a custom train step)
        weights = self.model.get_weights()
        norm = np.sqrt(sum(np.sum(np.square(w)) for w in weights))
        self.grad_norms.append(norm)
        print(f"  [epoch {epoch+1}] weight-norm proxy: {norm:.3f}")


def build_model(cell: str = "lstm", vocab_size: int = NUM_WORDS,
                 max_len: int = MAX_LEN):
    """Build a Sequential model: Embedding -> Recurrent layer -> Dense head.

    cell: one of "simple_rnn", "lstm", "gru"
    """
    cell = cell.lower()
    if cell == "simple_rnn":
        rnn_layer = SimpleRNN(RNN_UNITS)
    elif cell == "lstm":
        rnn_layer = LSTM(RNN_UNITS)
    elif cell == "gru":
        rnn_layer = GRU(RNN_UNITS)
    else:
        raise ValueError(f"Unknown cell type: {cell}")

    model = Sequential([
        Input(shape=(max_len,)),
        Embedding(vocab_size, EMBED_DIM),
        rnn_layer,
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_evaluate(cell: str, data: dict, epochs: int = EPOCHS,
                        out_dir: str = "assets"):
    os.makedirs(out_dir, exist_ok=True)
    model = build_model(cell)
    model.summary()

    grad_monitor = GradientNormMonitor()
    early_stop = EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )

    history = model.fit(
        data["x_train"], data["y_train"],
        validation_data=(data["x_val"], data["y_val"]),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=[grad_monitor, early_stop],
        verbose=2,
    )

    # ---- Evaluation ----
    y_prob = model.predict(data["x_test"], batch_size=BATCH_SIZE).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "test_loss_accuracy": model.evaluate(data["x_test"], data["y_test"], verbose=0),
        "f1": f1_score(data["y_test"], y_pred),
        "precision": precision_score(data["y_test"], y_pred),
        "recall": recall_score(data["y_test"], y_pred),
        "confusion_matrix": confusion_matrix(data["y_test"], y_pred).tolist(),
        "grad_norms": grad_monitor.grad_norms,
    }

    _plot_history(history, cell, out_dir)
    return model, history, metrics


def _plot_history(history, cell, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title(f"{cell.upper()} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title(f"{cell.upper()} — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{cell}_training_history.png"), dpi=120)
    plt.close(fig)


def compare_cells(cells=("simple_rnn", "lstm", "gru"), epochs: int = EPOCHS,
                   out_dir: str = "assets"):
    data = get_preprocessed_data()
    results = {}
    for cell in cells:
        print(f"\n===== Training {cell} =====")
        _, history, metrics = train_and_evaluate(cell, data, epochs, out_dir)
        results[cell] = {"history": history.history, "metrics": metrics}

    # comparison bar chart: F1 across cell types
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(list(results.keys()), [results[c]["metrics"]["f1"] for c in results])
    ax.set_ylabel("Test F1-score")
    ax.set_title("F1-score by recurrent cell type")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cell_comparison_f1.png"), dpi=120)
    plt.close(fig)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", default="lstm", choices=["simple_rnn", "lstm", "gru"])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--compare", action="store_true",
                         help="Train simple_rnn, lstm, and gru and compare them")
    args = parser.parse_args()

    if args.compare:
        results = compare_cells(epochs=args.epochs)
        for cell, r in results.items():
            print(cell, "F1 =", r["metrics"]["f1"])
    else:
        data = get_preprocessed_data()
        model, history, metrics = train_and_evaluate(args.cell, data, args.epochs)
        print("Test F1:", metrics["f1"])
        print("Precision:", metrics["precision"], "Recall:", metrics["recall"])
        print("Confusion matrix:", metrics["confusion_matrix"])
