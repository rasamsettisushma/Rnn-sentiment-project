"""
preprocess.py
--------------
Preprocessing pipeline for the IMDB movie review sentiment-analysis task.

The Keras `imdb` dataset ships pre-tokenized: every review is already a list
of integer word IDs (ranked by frequency). This module wraps that dataset
with the two preprocessing steps every RNN pipeline needs:

    1. Vocabulary capping   -> keep only the top `num_words` most frequent tokens
    2. Sequence padding      -> force every review to the same length `max_len`
                                 (truncate long reviews, zero-pad short ones)

It also exposes `decode_review()` so you can turn an integer sequence back
into readable English for sanity-checking, and a small CLI so the script
can be run standalone (`python src/preprocess.py`) to cache a ready-to-train
.npz file.
"""

import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---- Config -----------------------------------------------------------
NUM_WORDS = 10000   # vocabulary size: keep only the 10k most frequent words
MAX_LEN = 200        # pad/truncate every review to this many tokens
PAD_TYPE = "post"    # pad zeros at the end of short sequences
TRUNC_TYPE = "post"  # cut from the end of long sequences


def load_raw_data(num_words: int = NUM_WORDS):
    """Load the IMDB dataset as integer-encoded sequences.

    Returns (x_train, y_train), (x_test, y_test) where x_* are lists of
    variable-length integer arrays and y_* are 0/1 sentiment labels.
    """
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
    return (x_train, y_train), (x_test, y_test)


def pad_data(x, max_len: int = MAX_LEN):
    """Pad/truncate a list of integer sequences to a fixed length."""
    return pad_sequences(
        x, maxlen=max_len, padding=PAD_TYPE, truncating=TRUNC_TYPE
    )


def get_preprocessed_data(num_words: int = NUM_WORDS, max_len: int = MAX_LEN,
                           val_split: float = 0.1, seed: int = 42):
    """Full pipeline: load -> pad -> train/val/test split.

    Returns a dict with keys: x_train, y_train, x_val, y_val, x_test, y_test.
    """
    (x_train, y_train), (x_test, y_test) = load_raw_data(num_words)

    x_train = pad_data(x_train, max_len)
    x_test = pad_data(x_test, max_len)

    # carve a validation set out of training data
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(x_train))
    x_train, y_train = x_train[idx], y_train[idx]

    n_val = int(len(x_train) * val_split)
    x_val, y_val = x_train[:n_val], y_train[:n_val]
    x_train, y_train = x_train[n_val:], y_train[n_val:]

    return {
        "x_train": x_train, "y_train": y_train,
        "x_val": x_val, "y_val": y_val,
        "x_test": x_test, "y_test": y_test,
    }


def decode_review(sequence, index_from: int = 3):
    """Turn an integer-encoded review back into readable text (for debugging).

    Keras reserves 0=pad, 1=start, 2=unknown, so real word IDs start at 3.
    """
    word_index = imdb.get_word_index()
    reverse_index = {v + index_from: k for k, v in word_index.items()}
    reverse_index[0], reverse_index[1], reverse_index[2] = "<PAD>", "<START>", "<UNK>"
    return " ".join(reverse_index.get(i, "?") for i in sequence)


if __name__ == "__main__":
    data = get_preprocessed_data()
    print("Train:", data["x_train"].shape, data["y_train"].shape)
    print("Val:  ", data["x_val"].shape, data["y_val"].shape)
    print("Test: ", data["x_test"].shape, data["y_test"].shape)

    np.savez_compressed(
        "preprocessed_imdb.npz",
        x_train=data["x_train"], y_train=data["y_train"],
        x_val=data["x_val"], y_val=data["y_val"],
        x_test=data["x_test"], y_test=data["y_test"],
    )
    print("Saved preprocessed_imdb.npz")
