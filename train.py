"""Train the sentiment network on IMDB and export model + metadata to ./artifacts."""
import json

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from common import ART, MAXLEN, VOCAB_SIZE


def build_model() -> keras.Model:
    return keras.Sequential(
        [
            layers.Input(shape=(MAXLEN,), name="tokens"),
            layers.Embedding(VOCAB_SIZE, 64, name="embedding"),
            layers.Bidirectional(layers.LSTM(32), name="bilstm"),
            layers.Dense(32, activation="relu", name="dense"),
            layers.Dropout(0.4, name="dropout"),
            layers.Dense(1, activation="sigmoid", name="sentiment"),
        ],
        name="sentiscope",
    )


def main(epochs: int = 6) -> None:
    ART.mkdir(exist_ok=True)
    (x_tr, y_tr), (x_te, y_te) = keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)
    x_tr = keras.utils.pad_sequences(x_tr, maxlen=MAXLEN)
    x_te = keras.utils.pad_sequences(x_te, maxlen=MAXLEN)

    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    hist = model.fit(
        x_tr, y_tr, validation_split=0.15, epochs=epochs, batch_size=128, callbacks=[stop], verbose=2
    )

    loss, acc = model.evaluate(x_te, y_te, batch_size=512, verbose=0)
    pred = (model.predict(x_te, batch_size=512, verbose=0).ravel() >= 0.5).astype(int)
    y = np.asarray(y_te)
    cm = [
        [int(((pred == 0) & (y == 0)).sum()), int(((pred == 1) & (y == 0)).sum())],
        [int(((pred == 0) & (y == 1)).sum()), int(((pred == 1) & (y == 1)).sum())],
    ]

    layer_rows = []
    for lyr in model.layers:
        try:
            shape = str(tuple(lyr.output.shape))
        except Exception:
            shape = "-"
        layer_rows.append([lyr.name, lyr.__class__.__name__, shape, int(lyr.count_params())])

    word_index = keras.datasets.imdb.get_word_index()
    vocab = {w: r + 3 for w, r in word_index.items() if r + 3 < VOCAB_SIZE}

    metrics = {
        "test_acc": float(acc),
        "test_loss": float(loss),
        "params": int(model.count_params()),
        "epochs": len(hist.history["loss"]),
        "train_size": int(len(x_tr)),
        "test_size": int(len(x_te)),
        "vocab_size": VOCAB_SIZE,
        "maxlen": MAXLEN,
        "history": {k: [float(v) for v in vals] for k, vals in hist.history.items()},
        "confusion": cm,
        "layers": layer_rows,
    }
    model.save(ART / "model.keras")
    (ART / "vocab.json").write_text(json.dumps(vocab))
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Done. Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
