import wfdb

# CHANGE PATH
BASE_PATH = r"data/ptb-xl"

def load_ecg(record_relative_path):
    record = wfdb.rdsamp(f"{BASE_PATH}/{record_relative_path}")
    signal = record[0]   # ECG signal
    return signal

if __name__ == "__main__":
    # example ECG file
    sample_signal = load_ecg("records100/13000/13958_lr")
    print("ECG shape:", sample_signal.shape)
