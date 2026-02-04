import wfdb
import numpy as np
from filter_data import load_and_filter

# Dataset base path
BASE_PATH = r"data/ptb-xl"

def build_dataset(max_per_class=400):
    """
    Build a balanced ECG dataset for CNN training.
    max_per_class=400 → 5 classes × 400 = 2000 samples
    """

    # Load and filter metadata (single-label ECGs)
    df = load_and_filter()

    # -------------------------------
    # BALANCE THE DATASET (IMPORTANT)
    # -------------------------------
    balanced_df = (
        df.groupby("label")
          .apply(lambda x: x.sample(
              n=min(len(x), max_per_class),
              random_state=42
          ))
          .reset_index(drop=True)
    )

    print("Balanced class distribution:")
    print(balanced_df["label"].value_counts())

    X = []
    y = []

    # -------------------------------
    # LOAD ECG SIGNALS
    # -------------------------------
    for _, row in balanced_df.iterrows():
        record_path = f"{BASE_PATH}/{row['filename_lr']}"

        # Read ECG (1000 samples, 12 leads)
        signal, _ = wfdb.rdsamp(record_path)

        # Ensure fixed length (safety)
        signal = signal[:1000, :12]

        # Normalize per record
        signal = (signal - np.mean(signal)) / np.std(signal)

        X.append(signal)
        y.append(row["label"])

    # Convert to NumPy arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    print("Final dataset shapes:")
    print("X shape:", X.shape)  # (samples, 1000, 12)
    print("y shape:", y.shape)

    # Save dataset
    np.save("backend/ml/X.npy", X)
    np.save("backend/ml/y.npy", y)

    print("Dataset saved successfully")

if __name__ == "__main__":
    build_dataset()
