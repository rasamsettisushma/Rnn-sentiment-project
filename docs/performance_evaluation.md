# Performance Evaluation — RNN vs. LSTM vs. GRU

## 1. Setup

All three models share the identical architecture except for the recurrent
cell: `Embedding(10000, 64) -> {SimpleRNN|LSTM|GRU}(64) -> Dropout(0.3) ->
Dense(1, sigmoid)`. Same optimizer (Adam, lr=1e-3), same batch size (128),
same 200-token sequence length, early stopping on validation loss.

Run `python src/rnn_model.py --compare` to reproduce the numbers below and
regenerate `assets/*_training_history.png` and `assets/cell_comparison_f1.png`.

## 2. Metrics

| Metric | SimpleRNN | LSTM | GRU |
|---|---|---|---|
| Test accuracy | *(fill in)* | *(fill in)* | *(fill in)* |
| Test F1-score | *(fill in)* | *(fill in)* | *(fill in)* |
| Precision | *(fill in)* | *(fill in)* | *(fill in)* |
| Recall | *(fill in)* | *(fill in)* | *(fill in)* |
| Epochs to converge | *(fill in)* | *(fill in)* | *(fill in)* |

(Paste in the numbers `rnn_model.py` prints to stdout, or read them off
`assets/cell_comparison_f1.png`.)

## 3. Vanishing / Exploding Gradients

`rnn_model.py` logs a weight-norm proxy every epoch via
`GradientNormMonitor`. Expected pattern:

- **SimpleRNN**: weight/gradient magnitude typically **shrinks** across
  training on longer sequences — the classic vanishing-gradient problem.
  Because `tanh` derivatives are ≤ 1 and the same recurrent weight matrix is
  multiplied at every timestep, repeated multiplication drives gradients
  toward zero over ~200 steps, so the network struggles to connect early
  words in a review (e.g. "was not") to the label.
- **LSTM**: the cell state has an (almost) linear path controlled by the
  **forget gate**, so gradients can flow through many timesteps largely
  unchanged. This is why LSTMs handle long reviews noticeably better than
  SimpleRNN in practice.
- **GRU**: uses a similar gating idea (update + reset gates) with fewer
  parameters than LSTM (no separate cell state), so it trains faster and
  often matches LSTM accuracy on datasets of this size.

## 4. Discussion — LSTM vs. Standard RNN Limitations

- **Standard RNN** limitations:
  - Vanishing gradients on sequences longer than a few dozen steps, which
    caps how much long-range context the model can use.
  - No explicit mechanism to "forget" irrelevant information or "remember"
    important information selectively — the hidden state is overwritten
    wholesale at each step.
  - In practice this shows up as lower F1/accuracy and noisier validation
    curves compared to LSTM/GRU on the same data.
- **LSTM / GRU** address this via gating:
  - Forget/input/output gates (LSTM) or update/reset gates (GRU) let the
    network learn *what to keep* and *what to discard* at each timestep.
  - This preserves gradient flow across long sequences, so sentiment cues
    that appear early in a review still influence the final prediction.
  - Trade-off: more parameters and compute per timestep than SimpleRNN
    (LSTM > GRU > SimpleRNN in parameter count), though still tractable at
    this dataset size.

## 5. Conclusion

*(Fill in once you have run `--compare`: state which cell type won on F1,
by how much, and whether the extra LSTM parameters over GRU were worth it
for this dataset.)*
