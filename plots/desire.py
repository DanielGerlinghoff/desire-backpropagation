import numpy as np
import csv
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from desire_pytorch import SnnModel, SnnLinear, SnnFlatten


# Dataset
dataset = datasets.MNIST("../data", train=False, transform=transforms.ToTensor())

# Constants
neurons = [784, 1600, 800, 10]
layers  = 3
samples = 400

# Average loss for layers and epochs
contrib = np.zeros((layers, samples, max(neurons)))

for sample in range(samples):
    print(f"Sample: {sample+1}")

    # Load model
    state_dict, hyper_pars = torch.load(f"desire_pytorch_{sample:05d}.pt").values()

    model = SnnModel(hyper_pars)
    model.load_state_dict(state_dict)
    model.eval()

    # Iterate through test dataset
    for (image_cnt, (image, label)) in enumerate(dataset):
        if image_cnt == 800: break
        output = torch.sum(model(image), dim=0).type(torch.int)
        model.backward(torch.tensor(label))

        layer = 0
        for mod in model.modules():
            if type(mod) is SnnFlatten:
                spikes_in = mod.spikes.sum(dim=0).numpy()

            if type(mod) is SnnLinear:
                spikes_out = mod.spikes.sum(dim=0).numpy()
                desire     = mod.desire[:, 1].mul(2).sub(1) * mod.desire[:, 0]
                desire     = desire.numpy()

                contribution = np.multiply(mod.weights.numpy().transpose(), desire * spikes_out).transpose()
                contribution = np.multiply(contribution, spikes_in)
                contribution = contribution.sum(axis=0)
                contrib[layer, sample, :neurons[layer]] += contribution

                spikes_in = spikes_out
                layer += 1

# Plot spike activity
output = np.empty((layers, max(neurons), samples))

fig, axs = plt.subplots(layers, 1, sharex=True)
for lay in range(layers):
    c = contrib[lay, :, :neurons[lay]].transpose()
    c_max = np.log(max(abs(c.min()), abs(c.max())))
    axs[lay].imshow(np.log(np.abs(c) + 1) * np.sign(c),
                    cmap="PRGn", interpolation="none",
                    vmin=-c_max, vmax=c_max,
                    aspect=0.2*samples/neurons[lay])
    output[lay, :neurons[lay]] = c

plt.tight_layout()
plt.savefig("output.png", dpi=200)
torch.save(output, "output.pt")
