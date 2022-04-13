import numpy as np
import csv
import matplotlib.pyplot as plt
from torchvision import datasets


# Read CSV file and dataset
file_spikes = open("spikes.csv", "r")
csv_spikes  = csv.reader(file_spikes)

dataset = datasets.MNIST("../data", train=False)
label = list()
for _, lab in dataset:
    label.append(lab)

# Constants
neurons = [1600, 800, 10]
layers  = 3
samples = len(label)
labels  = 10
epochs  = 80

# Spike activity for each layer and label
layer  = 0
sample = 0
epoch  = 0
spikes = np.zeros((epochs, labels, layers, max(neurons)))

for line in csv_spikes:
    # Skip input layer
    if layer == 0:
        layer += 1
        continue

    # Save spike activity
    line = np.array(line).astype(np.float)
    spikes[epoch, label[sample], layer-1, :neurons[layer-1]] += line

    # Increment counters
    if layer - 1 < layers - 1:
        layer += 1
    else:
        layer = 0
        if sample < samples - 1:
            sample += 1
        else:
            sample = 0
            if epoch < epochs - 1:
                epoch += 1
                print(f"Epoch: {epoch}")
            else:
                break

# Plot spike activity
from matplotlib.colors import Normalize
fig, axs = plt.subplots(layers, labels, sharex=True, sharey="row")
for lab in range(labels):
    for lay in range(layers):
        s = spikes[:, lab, lay, :neurons[lay]].transpose()
        s = np.log10(s + 1)
        axs[lay, lab].imshow(s, cmap="inferno", interpolation="none",
                                vmin=s.min(), vmax=s.max(),
                                aspect=1.5*epochs/neurons[lay])

fig.set_size_inches(10, 4)
plt.tight_layout()
plt.savefig("fig.png", dpi=200)
