import numpy as np
from sklearn.preprocessing import LabelEncoder

y = np.load("backend/ml/y.npy")

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

np.save("backend/ml/y_encoded.npy", y_encoded)
np.save("backend/ml/classes.npy", encoder.classes_)

print("Classes:", encoder.classes_)
