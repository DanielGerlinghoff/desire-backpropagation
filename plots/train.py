import numpy as np
import csv
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from desire_pytorch import SnnModel


# Build model and dataset
state_dict, hyper_pars = torch.load("../desire_pytorch_134.pt").values()

model = SnnModel(hyper_pars)
model.load_state_dict(state_dict)

dataset = datasets.MNIST("../data", train=False, transform=transforms.ToTensor())
labels  = 10
samples = len(dataset)

torch.manual_seed(4)

# Spike trains for each class
outputs = np.empty((labels, hyper_pars.tsteps, labels), dtype=np.int)
for l in range(labels):
    # Pick random sample
    label = None
    while label != l:
        idx = torch.randint(samples, (1,)).item()
        input, label = dataset[idx]

    # Generate spike train
    output = model(input).to(int).numpy()
    outputs[label] = output[1:]

# Plot spike activity
from matplotlib.colors import Normalize
fig, axs = plt.subplots(1, labels, sharey=True)
for lab in range(labels):
    for row in range(labels):
        o = outputs[lab, :, row]
        axs[lab].bar(np.arange(o.size), o * 0.8, width=0.5, bottom=labels-1-row)

fig.set_size_inches(10, 2)
plt.tight_layout()
plt.savefig("fig.png", dpi=200)
