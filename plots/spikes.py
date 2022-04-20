import numpy as np
import csv
import matplotlib.pyplot as plt
from torchvision import datasets


# Read CSV file and dataset
file_spikes = open("spikes.csv", "r")
csv_spikes  = csv.reader(file_spikes)

dataset = datasets.MNIST("../data", train=True)
label = list()
for _, lab in dataset:
    label.append(lab)

# Constants
neurons  = [1600, 800, 10]
layers   = 3
epochs   = 5
labels   = 10
interval = 100
samples  = int(len(label) / labels * epochs * 1.2)
seed     = 3

# Spike activity for each layer and label
layer  = 0
sample = np.zeros(labels, dtype=np.int)
spikes = np.zeros((samples, labels, layers, max(neurons)))

for cnt, line in enumerate(csv_spikes):
    cnt = (cnt // layers)

    # Save spike activity
    line = np.array(line).astype(np.float)
    lab  = label[cnt%len(label)]
    spikes[sample[lab], lab, layer, :neurons[layer]] = line

    # Increment counters
    if layer < layers - 1:
        layer += 1
    else:
        layer = 0
        sample[lab] += 1

        # Break after epochs
        if sample.sum() == len(label) * epochs:
            break

print(sample)

# Plot spike activity
np.random.seed(seed)
selection = [np.random.randint(n, size=20) for n in neurons[:-1]]
selection.append(np.arange(neurons[-1]))
output = np.empty((max([s.size for s in selection]), labels, layers, sample.min() // interval))

import torch
import torch.nn.functional as F
fig, axs = plt.subplots(layers, labels, sharex="col", sharey="row")
for lab in range(labels):
    for lay in range(layers):
        s = spikes[:sample.min(), lab, lay, selection[lay]].transpose()
        s = torch.from_numpy(s)[None, None].to(torch.float)
        k = torch.ones((1, interval)).div(interval)[None, None]
        s = F.conv2d(s, k, stride=(1, interval), padding="valid").squeeze()
        s = np.log10(s.numpy() + 1)
        axs[lay, lab].imshow(s, cmap="inferno", interpolation="none",
                                vmin=s.min(), vmax=s.max(),
                                aspect=1.0*sample.max()/len(selection[lay])/interval)
        output[:len(selection[lay]), lab, lay, :s.shape[1]] = s

fig.set_size_inches(10, 4)
plt.tight_layout()
plt.savefig("output.png", dpi=200)
torch.save(output, "output.pt")

file_spikes.close()
