import wfdb
import matplotlib.pyplot as plt

BASE_PATH = r"C:\Users\y21ac\Downloads\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"

record = wfdb.rdsamp(f"{BASE_PATH}/records100/13000/13958_lr")
signal = record[0]

plt.figure(figsize=(12, 4))
plt.plot(signal[:, 0])  # Lead I
plt.title("ECG Signal - Lead I")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()
