# Install required packages
!pip install pandas numpy scikit-learn tensorflow

# Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

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

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_tfidf = tfidf.fit_transform(X_train).toarray()
X_val_tfidf = tfidf.transform(X_val).toarray()

# -----------------------------
# Build Neural Network
# -----------------------------
model = Sequential([
    Dense(128, activation="relu", input_dim=X_train_tfidf.shape[1]),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])

# -----------------------------
# Train Model
# -----------------------------
model.fit(X_train_tfidf, y_train, validation_data=(X_val_tfidf, y_val), epochs=5, batch_size=32)

# -----------------------------
# Evaluate with F1 Score
# -----------------------------
y_val_pred = (model.predict(X_val_tfidf) > 0.5).astype(int)
print("Validation F1 Score:", f1_score(y_val, y_val_pred))

# -----------------------------
# Load Test Data & Predict
# -----------------------------
test = pd.read_csv("test.csv")
X_test_tfidf = tfidf.transform(test["customer_review"].astype(str)).toarray()
test_preds = (model.predict(X_test_tfidf) > 0.5).astype(int)

# -----------------------------
# Prepare Submission (exact format: customer_review, feedback)
# -----------------------------
submission = pd.DataFrame({
    "customer_review": test["customer_review"],
    "feedback": test_preds.flatten()  # 0 = negative, 1 = positive
})

submission.to_csv("submissions.csv", index=False)
submission.head()
