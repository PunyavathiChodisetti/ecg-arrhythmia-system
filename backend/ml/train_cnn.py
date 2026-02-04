import os
import numpy as np
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Dense,
    Dropout,
    Flatten,
    BatchNormalization
)

from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# -------------------------------
# PATHS (SAFE)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

print("X shape:", X.shape)   # (samples, 1000, 12)
print("y shape:", y.shape)

# One-hot encode labels
y_cat = to_categorical(y)

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# BUILD 1D CNN (KERAS 3 SAFE)
# -------------------------------
inputs = Input(shape=(1000, 12))

x = Conv1D(32, kernel_size=7, activation="relu")(inputs)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)

x = Conv1D(64, kernel_size=5, activation="relu")(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)

x = Conv1D(128, kernel_size=3, activation="relu")(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)

x = Flatten()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)

outputs = Dense(y_cat.shape[1], activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)

# -------------------------------
# COMPILE
# -------------------------------
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
    X_train,
    y_train,
    validation_split=0.1,
    epochs=30,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# -------------------------------
# EVALUATE
# -------------------------------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("✅ CNN Test Accuracy:", acc)

# -------------------------------
# SAVE MODEL & CLASSES
# -------------------------------
model.save(MODEL_PATH)
np.save(CLASSES_PATH, classes)

print("✅ CNN model saved at:", MODEL_PATH)
