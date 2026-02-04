import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# -------------------------------
# LOAD DATA
# -------------------------------
X = np.load("backend/ml/X.npy")
y = np.load("backend/ml/y_encoded.npy")

# Split validation set
_, X_val, _, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# LOAD LOGITS MODEL
# -------------------------------
model = tf.keras.models.load_model("backend/ml/ecg_cnn_logits_model")

# -------------------------------
# GET LOGITS
# -------------------------------
logits = model.predict(X_val, batch_size=32)
labels = y_val

# -------------------------------
# TEMPERATURE SCALING
# -------------------------------
temperature = tf.Variable(1.0, dtype=tf.float32)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

for epoch in range(200):
    with tf.GradientTape() as tape:
        scaled_logits = logits / temperature
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels,
                logits=scaled_logits
            )
        )

    grads = tape.gradient(loss, [temperature])
    optimizer.apply_gradients(zip(grads, [temperature]))

    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Loss: {loss.numpy():.4f} | Temp: {temperature.numpy():.4f}")

# -------------------------------
# SAVE TEMPERATURE
# -------------------------------
np.save("backend/ml/temperature.npy", temperature.numpy())
print("\n✅ Optimal temperature saved:", temperature.numpy())
