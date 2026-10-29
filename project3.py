import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from collections import Counter
from datasets import load_dataset
from tensorflow.keras import layers, ops, Model
from tensorflow.keras.callbacks import EarlyStopping


# 1. Load the WikiText-2 dataset
print("Loading dataset...")

dataset = load_dataset(
    "Salesforce/wikitext",
    "wikitext-2-raw-v1"
)

train_text = " ".join(dataset["train"]["text"][:10000])
val_text = " ".join(dataset["validation"]["text"][:2000])


# 2. Convert text into words
def tokenize(text):
    return re.findall(
        r"[a-z]+(?:'[a-z]+)?|\d+|[^\w\s]",
        text.lower()
    )


train_words = tokenize(train_text)
val_words = tokenize(val_text)

print("Training words:", len(train_words))
print("Validation words:", len(val_words))


# 3. Create vocabulary
word_counts = Counter(train_words)

vocab = ["<pad>", "<unk>"] + [
    word for word, count in word_counts.most_common(5000)
]

word_to_id = {
    word: index for index, word in enumerate(vocab)
}

id_to_word = {
    index: word for word, index in word_to_id.items()
}

VOCAB_SIZE = len(vocab)

print("Vocabulary size:", VOCAB_SIZE)


# 4. Convert words into numbers
def encode_words(words):
    return [
        word_to_id.get(word, word_to_id["<unk>"])
        for word in words
    ]


train_ids = encode_words(train_words)
val_ids = encode_words(val_words)


# 5. Create input and next-token target sequences
SEQ_LEN = 20


def make_examples(token_ids, limit):
    X = []
    y = []

    total = min(len(token_ids) - SEQ_LEN, limit)

    for i in range(total):
        # Input: current 20 tokens
        X.append(token_ids[i:i + SEQ_LEN])

        # Target: next token at every position
        y.append(token_ids[i + 1:i + SEQ_LEN + 1])

    return (
        np.array(X, dtype=np.int32),
        np.array(y, dtype=np.int32)
    )


X_train, y_train = make_examples(train_ids, 50000)
X_val, y_val = make_examples(val_ids, 5000)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)


# 6. Build the Transformer model
EMBED_DIM = 128
NUM_HEADS = 4

inputs = layers.Input(shape=(SEQ_LEN,), dtype="int32")

# Token embeddings
token_embeddings = layers.Embedding(
    input_dim=VOCAB_SIZE,
    output_dim=EMBED_DIM
)(inputs)

# Positional embeddings
positions = ops.zeros_like(inputs) + ops.arange(SEQ_LEN)

position_embeddings = layers.Embedding(
    input_dim=SEQ_LEN,
    output_dim=EMBED_DIM
)(positions)

# Add token and position embeddings
x = layers.Add()([token_embeddings, position_embeddings])
x = layers.Dropout(0.2)(x)

# Causal self-attention
attention_output = layers.MultiHeadAttention(
    num_heads=NUM_HEADS,
    key_dim=EMBED_DIM // NUM_HEADS,
    dropout=0.1
)(
    x,
    x,
    use_causal_mask=True
)

x = layers.Add()([x, attention_output])
x = layers.LayerNormalization()(x)

# Feed-forward network
ffn = layers.Dense(256, activation="relu")(x)
ffn = layers.Dropout(0.2)(ffn)
ffn = layers.Dense(EMBED_DIM)(ffn)

x = layers.Add()([x, ffn])
x = layers.LayerNormalization()(x)

# Predict a token at every sequence position
outputs = layers.Dense(VOCAB_SIZE)(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    ),
    metrics=["accuracy"]
)

model.summary()


# 7. Train with early stopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=32,
    callbacks=[early_stop]
)


# 8. Save the trained model
model.save("next_token_model.keras")

print("Model saved successfully!")


# 9. Predict the next word
def predict_next_words(text, top_n=5):
    words = tokenize(text)

    if not words:
        print("Please enter some words.")
        return

    # Keep only the latest 20 words
    words = words[-SEQ_LEN:]

    token_ids = encode_words(words)

    # Add padding at the end if the context is short
    actual_length = len(token_ids)
    token_ids = token_ids + (
        [word_to_id["<pad>"]] * (SEQ_LEN - actual_length)
    )

    input_data = np.array([token_ids], dtype=np.int32)

    # Get predictions for all positions
    logits = model.predict(input_data, verbose=0)

    # Use the prediction after the final real input word
    next_token_logits = logits[0, actual_length - 1]

    probabilities = tf.nn.softmax(
        next_token_logits
    ).numpy()

    ranked_ids = np.argsort(probabilities)[::-1]

    print("\nInput:", text)
    print("Top next-word predictions:")

    count = 0

    for token_id in ranked_ids:
        word = id_to_word[int(token_id)]

        if word in ("<pad>", "<unk>"):
            continue

        print(
            f"{word}: {probabilities[token_id] * 100:.2f}%"
        )

        count += 1

        if count == top_n:
            break


# 10. Test the model
predict_next_words("the history of")


# 11. Plot training and validation accuracy
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Transformer Next-Token Prediction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_accuracy.png")
plt.show()
