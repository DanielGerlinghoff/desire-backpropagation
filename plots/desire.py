import numpy as np
import csv
import matplotlib.pyplot as plt


# Read CSV files
file_desire = open("desire.csv", "r")
csv_desire  = csv.reader(file_desire)
file_spikes = open("spikes.csv", "r")
csv_spikes  = csv.reader(file_spikes)
file_weight = open("weight.csv", "r")
csv_weight  = csv.reader(file_weight)

# Constants
neurons = [784, 1600, 800, 10]
layers  = 3
samples = 10000
labels  = 10
epochs  = 40

# Average loss for layers and epochs
layer   = 0
sample  = 0
epoch   = 0
contrib = np.zeros((epochs, layers, max(neurons)))

while epoch < epochs:
    # Load weights for epoch
    weight = [None] * layers
    for lay in range(layers):
        weight[lay] = np.empty(neurons[lay:lay+2])
        for neu in range(neurons[lay]):
            weight[lay][neu] = np.array(next(csv_weight)).astype(np.float)

    # Iterate over samples
    while sample < samples:
        while layer < layers:
            spikes = np.array(next(csv_spikes)).astype(np.int)
            desire = np.array(next(csv_desire)).astype(np.int)

            contribution = np.multiply(weight[layer], desire).transpose()
            #contribution = np.multiply(contribution, spikes)
            contribution = contribution.sum(axis=0)
            contrib[epoch, layer, :neurons[layer]] += contribution

            layer += 1

        # Discard last layer's spikes
        spikes = next(csv_spikes)

        layer = 0
        sample += 1

    sample = 0
    epoch += 1
    print(f"Epoch: {epoch}")

# Plot spike activity
fig, axs = plt.subplots(layers, 1, sharex=True)
c_max = np.log2(max(abs(contrib.min()), abs(contrib.max())))
for lay in range(layers):
    c = contrib[:, lay, :neurons[lay]].transpose()
    axs[lay].imshow(np.log(np.abs(c) + 1) * np.sign(c),
                    cmap="PRGn", interpolation="none",
                    vmin=-c_max, vmax=c_max,
                    aspect=epochs/4/neurons[lay])

plt.tight_layout()
plt.savefig("fig.png", dpi=200)
