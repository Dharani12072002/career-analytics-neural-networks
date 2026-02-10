# ==========================================
# Student Performance Classification using NN
# Classes: Fail / Pass / Distinction
# ==========================================

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "StudentsPerformance.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print(df.head())

# -------------------------------
# 2. Create Student_ID (Identifier)
# -------------------------------
df["Student_ID"] = np.arange(1, len(df) + 1)

# -------------------------------
# 3. Create Target Labels
# -------------------------------
df["avg_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)

def label_perf(x):
    if x < 50:
        return "Fail"
    elif x <= 75:
        return "Pass"
    else:
        return "Distinction"

df["label"] = df["avg_score"].apply(label_perf)

print("\nClass distribution:\n", df["label"].value_counts())

# Encode labels
le = LabelEncoder()
df["y"] = le.fit_transform(df["label"])  # 0,1,2

# -------------------------------
# 4. Prepare Features and Target
# -------------------------------
ids = df["Student_ID"]  # keep IDs for saving results

X = pd.get_dummies(df.drop(columns=["label", "y", "avg_score", "Student_ID"]), drop_first=True)
y = df["y"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 5. Train-Test Split (including IDs)
# -------------------------------
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X_scaled, y, ids, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# -------------------------------
# 6. Build Neural Network Model
# -------------------------------
model = Sequential()

# Input layer + Hidden layer 1
model.add(Dense(64, activation="relu", input_shape=(X_train.shape[1],)))

# Hidden layer 2
model.add(Dense(32, activation="relu"))

# Dropout for regularization
model.add(Dropout(0.3))

# Output layer (3 classes)
model.add(Dense(3, activation="softmax"))

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------
# 7. Train Model
# -------------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# -------------------------------
# 8. Evaluate Model
# -------------------------------
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

acc = accuracy_score(y_test, y_pred)
print("\nTest Accuracy:", acc)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# -------------------------------
# 9. Save Test Results with Identifier
# -------------------------------
results_df = pd.DataFrame({
    "Student_ID": id_test.values,
    "Actual_Label": le.inverse_transform(y_test),
    "Predicted_Label": le.inverse_transform(y_pred)
})

output_file = os.path.join(OUTPUT_DIR, "student_performance_test_results.csv")
results_df.to_csv(output_file, index=False)

print(f"\n✅ Test results saved at: {output_file}")
