import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Dense,
    Dropout, Flatten, BatchNormalization
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------------
# PATHS (SAFE)
# -------------------------------
BASE_DIR = os.path.dirname(__file__)

X_PATH = os.path.join(BASE_DIR, "X.npy")
Y_PATH = os.path.join(BASE_DIR, "y_encoded.npy")
CLASSES_PATH = os.path.join(BASE_DIR, "classes.npy")
MODEL_PATH = os.path.join(BASE_DIR, "ecg_cnn_model.keras")

# -------------------------------
# LOAD DATA
# -------------------------------
X = np.load(X_PATH)
y = np.load(Y_PATH)
classes = np.load(CLASSES_PATH, allow_pickle=True)

print("X shape:", X.shape)
print("y shape:", y.shape)

# One-hot encode labels
y_cat = to_categorical(y)

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# BUILD 1D CNN
# -------------------------------
model = Sequential([
    Input(shape=(1000, 12)),

    Conv1D(32, kernel_size=7, activation="relu"),
    BatchNormalization(),
    MaxPooling1D(2),

    Conv1D(64, kernel_size=5, activation="relu"),
    BatchNormalization(),
    MaxPooling1D(2),

    Conv1D(128, kernel_size=3, activation="relu"),
    BatchNormalization(),
    MaxPooling1D(2),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(y_cat.shape[1], activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------
# TRAIN
# -------------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=30,
    batch_size=32,
    callbacks=[early_stop]
)

# -------------------------------
# EVALUATE
# -------------------------------
loss, acc = model.evaluate(X_test, y_test)
print("CNN Test Accuracy:", acc)

# -------------------------------
# SAVE MODEL & CLASSES
# -------------------------------
model.save(MODEL_PATH)
np.save(CLASSES_PATH, classes)

print("✅ CNN model saved at:", MODEL_PATH)
