import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import itertools

# Define layers
class snn_linear(nn.Module):
    def __init__(self, idx, neu_in, neu_out, tsteps):
        super().__init__()
        self.idx          = idx
        self.neu_pre_cnt  = neu_in
        self.neu_post_cnt = neu_out
        self.tstep_cnt    = tsteps

        # Network parameters
        if train_ntest:
            weights = np.random.randn(neu_in, neu_out).astype(np.float32) * np.sqrt(2 / neu_in)
        else:
            weights = np.load(f"model/weights_{idx}.npy")
        self.weights = nn.Parameter(torch.from_numpy(weights))
        self.weights.requires_grad = False

        self.spikes = torch.zeros((tsteps + 1, neu_out), dtype=torch.bool)
        self.mempot = torch.zeros(neu_in, dtype=torch.float32)
        self.traces = torch.zeros((tsteps + 1, neu_in), dtype=torch.float32)
        self.desire = torch.zeros((neu_out, 2), dtype=torch.bool)

    def forward(self, spikes_in, tstep, traces=False):
        spikes_out = torch.zeros(self.neu_post_cnt, dtype=torch.bool)

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

        self.spikes[tstep+1] = spikes_out

        # Generate traces
        if traces: self.gen_traces(spikes_in, tstep)

    def backward(self, desire_in):
        desire_out  = torch.zeros((self.neu_pre_cnt, 2), dtype=torch.bool)

        for neu_pre in range(self.neu_pre_cnt):
            desire_sum = 0
            for neu_post in range(self.neu_post_cnt):
                if desire_in[neu_post][0]:
                    if desire_in[neu_post][1]:
                        error = 1 - torch.sum(self.spikes[:, neu_post]) / self.tstep_cnt
                        desire_sum += self.weights[neu_pre][neu_post] * error
                    else:
                        error = torch.sum(self.spikes[:, neu_post]) / self.tstep_cnt
                        desire_sum -= self.weights[neu_pre][neu_post] * error
            desire_out[neu_pre][0] = abs(desire_sum) >= desire_thres["hidden"]
            desire_out[neu_pre][1] = desire_sum > 0

        return desire_out

    def gen_traces(self, spikes_in, tstep):
        for neu_pre in range(self.neu_pre_cnt):
            trace = self.traces[tstep][neu_pre]
            self.traces[tstep+1][neu_pre] = ((trace * 2 ** decay) - trace) / 2 ** decay

            if spikes_in[neu_pre]:
                self.traces[tstep+1][neu_pre] += 1

    def reset(self):
        self.spikes = torch.zeros((self.tstep_cnt + 1, self.neu_post_cnt), dtype=torch.bool)
        self.mempot = torch.zeros(self.neu_pre_cnt, dtype=torch.float32)
        self.traces = torch.zeros((self.tstep_cnt + 1, self.neu_pre_cnt), dtype=torch.float32)
        self.desire = torch.zeros((self.neu_post_cnt, 2), dtype=torch.bool)

class snn_input(nn.Module):
    def __init__(self, neu_in, tsteps):
        super().__init__()
        self.neu_in_cnt = neu_in
        self.tstep_cnt  = tsteps

        # Network parameters
        self.spikes = torch.zeros((tsteps + 1, neu_in), dtype=torch.bool)
        self.mempot = torch.zeros(neu_in, dtype=torch.float32)

    def forward(self, image, tstep):
        spikes_out = torch.zeros(self.neu_in_cnt, dtype=torch.bool)

        # Generate input spike train from image
        for neu_in in range(self.neu_in_cnt):
            self.mempot[neu_in] += image[neu_in]
            if self.mempot[neu_in] >= mempot_thres:
                spikes_out[neu_in] = True
                self.mempot[neu_in] -= mempot_thres

        self.spikes[tstep] = spikes_out

    def reset(self):
        self.spikes = torch.zeros((self.tstep_cnt + 1, self.neu_in_cnt), dtype=torch.bool)
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
        self.inp  = snn_input(neurons[0], tsteps)
        self.lin1 = snn_linear(0, neurons[0], neurons[1], tsteps)
        self.lin2 = snn_linear(1, neurons[1], neurons[2], tsteps)
        self.lin3 = snn_linear(2, neurons[2], neurons[3], tsteps)

    def forward(self, image):
        # Reset membrane potentials
        self.inp.reset()
        self.lin1.reset()
        self.lin2.reset()
        self.lin3.reset()

        # Process spikes
        image = self.flat(image)
        for tstep in range(self.tstep_cnt):
            self.inp(image, tstep)
            self.lin1(self.inp.spikes[tstep], tstep, traces=train_ntest)
            self.lin2(self.lin1.spikes[tstep], tstep, traces=train_ntest)
            self.lin3(self.lin2.spikes[tstep], tstep, traces=train_ntest)

        return self.lin3.spikes

    def backward(self, label):
        # Desire of output layer
        for neu_out in range(self.neuron_cnt[-1]):
            if neu_out == label:
                self.lin3.desire[neu_out][0] = 1 - torch.sum(self.lin3.spikes[:,neu_out]) / self.tstep_cnt
            else:
                self.lin3.desire[neu_out][0] = torch.sum(self.lin3.spikes[:,neu_out]) / self.tstep_cnt
            self.lin3.desire[neu_out][0] = self.lin3.desire[neu_out][0] >= desire_thres["output"]
            self.lin3.desire[neu_out][1] = neu_out == label

        # Backpropagate desire
        self.lin2.desire = self.lin3.backward(self.lin3.desire)
        self.lin1.desire = self.lin2.backward(self.lin2.desire)

# Define optimizer
class snn_optim(torch.optim.Optimizer):
    def __init__(self, params):
        defaults = dict()

        super().__init__(params, defaults)

    def step(self, closure=None):
        pass


if __name__ == "__main__":
    # Network configuration
    neurons   = (784, 512, 256, 10)
    layer_cnt = len(neurons) - 1
    tstep_cnt = 20
    image_cnt = 2000
    epoch_cnt = 1

    train_ntest   = True
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
            model.backward(label)
            if debug:
                print(np.array(torch.sum(spikes, dim=0)))

pass
