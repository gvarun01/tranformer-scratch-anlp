# Install required packages
!pip install pandas numpy scikit-learn tensorflow

# Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# Load Training Data
# -----------------------------
data = pd.read_csv("train.csv")
data.head()

# -----------------------------
# Preprocessing + Train/Val Split
# -----------------------------
X = data["customer_review"].astype(str)
y = data["feedback"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenization + Padding
max_words = 10000  # vocabulary size
max_len = 100      # max length of sequence

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_val_pad = pad_sequences(X_val_seq, maxlen=max_len, padding='post', truncating='post')

# -----------------------------
# Build LSTM Model
# -----------------------------
model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])

# -----------------------------
# Train Model with EarlyStopping
# -----------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

model.fit(
    X_train_pad, y_train,
    validation_data=(X_val_pad, y_val),
    epochs=10,
    batch_size=32,
    callbacks=[early_stop]
)

# -----------------------------
# Evaluate with F1 Score
# -----------------------------
y_val_pred = (model.predict(X_val_pad) > 0.5).astype(int)
print("Validation F1 Score:", f1_score(y_val, y_val_pred))

# -----------------------------
# Load Test Data & Predict
# -----------------------------
test = pd.read_csv("test.csv")
X_test_seq = tokenizer.texts_to_sequences(test["customer_review"].astype(str))
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')
test_preds = (model.predict(X_test_pad) > 0.5).astype(int)

# -----------------------------
# Prepare Submission (exact format: customer_review, feedback)
# -----------------------------
submission = pd.DataFrame({
    "customer_review": test["customer_review"],
    "feedback": test_preds.flatten()  # 0 = negative, 1 = positive
})

submission.to_csv("submissions.csv", index=False)
submission.head()
