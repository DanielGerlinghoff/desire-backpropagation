import numpy as np
import csv
import matplotlib.pyplot as plt


# CSV files
file_error  = open("error.csv", "r")
csv_error   = csv.reader(file_error)
file_output = open("output.csv", "w")
csv_output  = csv.writer(file_output)

# Loss function and constants
loss = lambda error: np.sum(error ** 2)

neurons = [1600, 800, 10]
layers  = 3
samples = 10000
epochs  = 150

# Average loss for layers and epochs
layer  = 0
sample = 0
epoch  = 0
losses = np.zeros((epochs, layers))

for line in csv_error:
    if line:
        line = np.array(line).astype(np.float)
        losses[epoch, layer] += loss(line) / neurons[layer] / samples

    if layer < layers - 1:
        layer += 1
    else:
        layer = 0
        if sample < samples - 1:
            sample += 1
        else:
            sample = 0
            print(f"Epoch: {epoch+1:3d}")
            if epoch < epochs - 1:
                epoch += 1
            else:
                break

# Plot losses
plt.plot(losses[:, 0], label="0", linestyle="None", marker=".")
plt.plot(losses[:, 1], label="1", linestyle="None", marker=".")
plt.plot(losses[:, 2], label="2", linestyle="None", marker=".")
plt.legend()
plt.savefig("output.png")

# Save output
for e in range(epochs):
    csv_output.writerow(losses[e].tolist())

file_error.close()
file_output.close()
