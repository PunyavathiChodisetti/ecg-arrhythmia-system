import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load saved dataset
X = np.load("backend/ml/X.npy")
y = np.load("backend/ml/y.npy")

print("Loaded X:", X.shape)
print("Loaded y:", y.shape)

# Encode labels (e.g., NORM → 0)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("Train samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])

# Flatten ECG signals for ML model
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_flat, y_train)

# Evaluate
y_pred = model.predict(X_test_flat)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save model and label encoder
joblib.dump(model, "backend/ml/ecg_model.pkl")
joblib.dump(le, "backend/ml/label_encoder.pkl")

print("Model saved successfully")
