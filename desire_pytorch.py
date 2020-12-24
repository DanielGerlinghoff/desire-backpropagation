import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import itertools

# Define layers
class snn_linear(nn.Module):
    def __init__(self, idx, neu_in, neu_out):
        super().__init__()
        self.neu_pre_cnt  = neu_in
        self.neu_post_cnt = neu_out

        # Network parameters
        if train_ntest:
            weights = np.random.randn(neu_in, neu_out).astype(np.float32) * np.sqrt(2 / neu_in)
        else:
            weights = np.load(f"model/weights_{idx}.npy")
        self.weights = nn.Parameter(torch.from_numpy(weights))
        self.weights.requires_grad = False

        self.mempot = torch.zeros(neu_in, dtype=torch.float32)
        self.desire = torch.zeros((neu_out, 2), dtype=torch.bool)

    def forward(self, spikes_in):
        spikes_out = torch.zeros(self.neu_post_cnt, dtype=torch.bool)
        traces_out = torch.zeros(self.neu_post_cnt, dtype=torch.float32)

        for neu_post in range(self.neu_post_cnt):
            # Update membrane potential
            for neu_pre in range(self.neu_pre_cnt):
                if spikes_in[neu_pre]:
                    self.mempot[neu_post] += self.weights[neu_pre][neu_post]

            # Calculate output spikes and decay membrane potential
            if self.mempot[neu_post] >= mempot_thres:
                spikes_out[neu_post] = True
                self.mempot[neu_post] -= mempot_thres
            else:
                mempot_old = self.mempot[neu_post]
                self.mempot[neu_post] = ((mempot_old * 2 ** decay) - mempot_old) / 2 ** decay

        return spikes_out, traces_out

    def reset(self):
        self.mempot = torch.zeros(self.neu_pre_cnt, dtype=torch.float32)
        self.desire = torch.zeros((self.neu_post_cnt, 2), dtype=torch.bool)

class snn_input(nn.Module):
    def __init__(self, neu_in):
        super().__init__()
        self.neu_in_cnt = neu_in

        # Network parameters
        self.mempot = torch.zeros(neu_in, dtype=torch.float32)

    def forward(self, image):
        spikes_out = torch.zeros(self.neu_in_cnt, dtype=torch.bool)
        traces_out = torch.zeros(self.neu_in_cnt, dtype=torch.float32)

        # Generate input spike train from image
        for neu_in in range(self.neu_in_cnt):
            self.mempot[neu_in] += image[neu_in]
            if self.mempot[neu_in] >= mempot_thres:
                spikes_out[neu_in] = True
                self.mempot[neu_in] -= mempot_thres

        return spikes_out, traces_out

    def reset(self):
        self.mempot = torch.zeros(self.neu_in_cnt, dtype=torch.float32)

# Define network
class snn_model(nn.Module):
    def __init__(self, neurons, tsteps):
        super().__init__()
        self.neuron_cnt = neurons
        self.layer_cnt  = len(neurons) - 1
        self.tstep_cnt  = tsteps

        # Layers
        np.random.seed(0)

        self.flat = nn.Flatten(0, -1)
        self.inp  = snn_input(neurons[0])
        self.lin1 = snn_linear(0, neurons[0], neurons[1])
        self.lin2 = snn_linear(1, neurons[1], neurons[2])
        self.lin3 = snn_linear(2, neurons[2], neurons[3])

        # Data arrays
        self.spikes, self.traces = ([], [])
        for layer in range(self.layer_cnt + 1):
            self.spikes.append(torch.zeros((tstep_cnt + 1, neurons[layer]), dtype=torch.bool))
            self.traces.append(torch.zeros((tstep_cnt + 1, neurons[layer]), dtype=torch.float32))
    
    def forward(self, image):
        # Reset membrane potentials
        self.inp.reset()
        self.lin1.reset()
        self.lin2.reset()
        self.lin3.reset()

        # Process spikes
        image = self.flat(image)
        for tstep in range(self.tstep_cnt):
            self.spikes[0][tstep], self.traces[0][tstep]     = self.inp(image)
            self.spikes[1][tstep+1], self.traces[1][tstep+1] = self.lin1(self.spikes[0][tstep])
            self.spikes[2][tstep+1], self.traces[2][tstep+1] = self.lin2(self.spikes[1][tstep])
            self.spikes[3][tstep+1], self.traces[3][tstep+1] = self.lin3(self.spikes[2][tstep])

        return self.spikes

if __name__ == "__main__":
    # Network configuration
    neurons   = (784, 512, 256, 10)
    layer_cnt = len(neurons) - 1
    tstep_cnt = 20
    image_cnt = 2000
    epoch_cnt = 1

    train_ntest   = False
    debug         = True
    debug_period  = 2000
    mempot_thres  = 1
    learning_rate = 1.e-3 / tstep_cnt
    decay         = 1
    desire_thres  = {"hidden": 0.1, "output": 0.1}

    model = snn_model(neurons, tstep_cnt)

    # Iterate over MNIST images
    dataset = datasets.MNIST("data", train=train_ntest, download=True, transform=transforms.ToTensor())
    for epoch in range(epoch_cnt):
        if debug: print(f"Epoch: {epoch}")

        for (image_idx, (image, label)) in enumerate(itertools.islice(dataset, image_cnt)):
            if debug: print(f"Image {image_idx}: {label}")

            # Forward pass
            spikes = model(image)
            if debug:
                print(np.array(torch.sum(spikes[-1], dim=0)))

pass
