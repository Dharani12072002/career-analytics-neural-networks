import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "careerChangePredictionDataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# -------------------------------
# 1) LOAD DATASET
# -------------------------------
df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)
print(df.head())

# -------------------------------
# 2) BASIC DATA CLEANING
# -------------------------------
df = df.dropna().reset_index(drop=True)
print("\nAfter cleaning shape:", df.shape)

# -------------------------------
# 3) CREATE AN IDENTIFIER COLUMN
# -------------------------------
df["Record_ID"] = df.index + 1   # Simple ID: 1, 2, 3, ...

# -------------------------------
# 4) CREATE CAREER TRANSITION DIFFICULTY INDEX
# -------------------------------
df['difficulty_index'] = (
    0.4 * df['Years of Experience'] +
    0.3 * (100 - df['Job Satisfaction']) +
    0.2 * (100 - df['Work-Life Balance']) +
    0.1 * df['Technology Adoption']
)

print("\nSample difficulty index:")
print(df[['Record_ID',
          'Years of Experience',
          'Job Satisfaction',
          'Work-Life Balance',
          'Technology Adoption',
          'difficulty_index']].head())

# -------------------------------
# 5) DEFINE FEATURES (X), TARGET (y), IDS
# -------------------------------
features = [
    'Years of Experience',
    'Job Satisfaction',
    'Work-Life Balance',
    'Technology Adoption'
]

X = df[features]
y = df['difficulty_index']
ids = df['Record_ID']

# -------------------------------
# 6) TRAIN-TEST SPLIT (KEEP IDS)
# -------------------------------
X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X, y, ids, test_size=0.2, random_state=42
)

print("\nTraining samples:", X_train.shape)
print("Testing samples:", X_test.shape)

# -------------------------------
# 7) FEATURE SCALING
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 8) BUILD NEURAL NETWORK MODEL
# -------------------------------
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.2),

    Dense(32, activation='relu'),

    Dense(1, activation='linear')   # Regression output
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.summary()

# -------------------------------
# 9) TRAIN MODEL
# -------------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# -------------------------------
# 10) PLOT TRAINING LOSS
# -------------------------------
plt.figure(figsize=(7,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "training_vs_validation_loss.png"))
plt.close()

# -------------------------------
# 11) PREDICTION ON TEST DATA
# -------------------------------
y_pred = model.predict(X_test_scaled).flatten()

# -------------------------------
# 12) REGRESSION METRICS
# -------------------------------
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===== REGRESSION PERFORMANCE ====")
print("MAE  :", mae)
print("MSE  :", mse)
print("RMSE :", rmse)
print("R2   :", r2)

# -------------------------------
# 13) ERROR DISTRIBUTION PLOT
# -------------------------------
errors = y_test - y_pred

plt.figure(figsize=(7,5))
plt.hist(errors, bins=30)
plt.title("Error Distribution Plot")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "error_distribution.png"))
plt.close()

# -------------------------------
# 14) ACTUAL vs PREDICTED PLOT
# -------------------------------
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Difficulty Index")
plt.ylabel("Predicted Difficulty Index")
plt.title("Actual vs Predicted Difficulty Index")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "actual_vs_predicted.png"))
plt.close()

# -------------------------------
# 15) SAVE RESULTS WITH IDENTIFIER
# -------------------------------
results_df = pd.DataFrame({
    "Record_ID": ids_test.values,
    "Actual_Difficulty": y_test.values,
    "Predicted_Difficulty": y_pred,
    "Error": errors
})

output_excel_path = os.path.join(OUTPUT_DIR, "career_transition_results.xlsx")
results_df.to_excel(output_excel_path, index=False)

print("\nResults saved as:", output_excel_path)

# -------------------------------
# 16) SAMPLE PREDICTION (NEW PERSON)
# -------------------------------
new_data = pd.DataFrame({
    'Years of Experience': [5],
    'Job Satisfaction': [60],
    'Work-Life Balance': [70],
    'Technology Adoption': [20]
})

new_data_scaled = scaler.transform(new_data)
predicted_score = model.predict(new_data_scaled)

print("\nPredicted Career Transition Difficulty Index:",
      float(predicted_score))
