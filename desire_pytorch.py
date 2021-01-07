import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import math
import argparse

# Define layers
class snn_linear(nn.Module):
    def __init__(self, idx, neu_in, neu_out, hyp):
        super().__init__()
        self.idx          = idx
        self.neu_pre_cnt  = neu_in
        self.neu_post_cnt = neu_out
        self.hyp          = hyp
        self.tstep_cnt    = hyp.tsteps
        self.decay        = (2 ** hyp.mempot_decay - 1) / 2 ** hyp.mempot_decay

        # Network parameters
        self.weights = nn.Parameter(torch.randn(neu_out, neu_in))
        nn.init.kaiming_normal_(self.weights)
        self.weights.requires_grad = False

        # Loss function
        self.loss = nn.L1Loss(reduction="none")

        self.reset()

    def forward(self, spikes_in, tstep, traces=False):
        # Update membrane potential
        self.mempot.addmv_(self.weights, spikes_in.type(torch.float32))
        mempot_ge_thres = self.mempot.ge(self.hyp.mempot_thres)
        self.mempot.sub_(mempot_ge_thres.type(torch.int), alpha=self.hyp.mempot_thres)

        # Calculate output spikes
        self.spikes[tstep+1] = mempot_ge_thres

        # Decay membrane potential
        self.mempot.mul_(mempot_ge_thres.logical_not().mul(self.decay).add(mempot_ge_thres))

        # Generate traces
        if traces: self.gen_traces(spikes_in, tstep)

    def backward(self, desire_in):
        # Output error
        spikes_sum = torch.sum(self.spikes.type(torch.float32), dim=0)
        error = self.loss(desire_in[:, 1].type(torch.float32), spikes_sum.div(self.tstep_cnt))

        # Sum weights and errors
        sign = desire_in[:, 1].mul(2).sub(1)
        sign.mul_(desire_in[:, 0])

        desire_sum = torch.sum(self.weights.mul(torch.mul(error, sign).repeat(self.neu_pre_cnt, 1).t()), dim=0)

        # Desire of previous layer
        desire_out_0 = desire_sum.abs().ge(self.hyp.desire_thres["hid"])
        desire_out_1 = desire_sum.gt(0)
        desire_out   = torch.stack((desire_out_0, desire_out_1), dim=1)

        return desire_out

    def gen_traces(self, spikes_in, tstep):
        self.traces[tstep+1] = self.traces[tstep].mul(self.decay)
        self.traces[tstep+1].add_(spikes_in.type(torch.int))

    def reset(self):
        self.spikes = torch.zeros((self.tstep_cnt + 1, self.neu_post_cnt), dtype=torch.bool)
        self.mempot = torch.zeros(self.neu_pre_cnt, dtype=torch.float32)
        self.traces = torch.zeros((self.tstep_cnt + 1, self.neu_pre_cnt), dtype=torch.float32)
        self.desire = torch.zeros((self.neu_post_cnt, 2), dtype=torch.bool)

class snn_input(nn.Module):
    def __init__(self, neu_in, hyp):
        super().__init__()
        self.neu_in_cnt = neu_in
        self.hyp        = hyp
        self.tstep_cnt  = hyp.tsteps

        # Network parameters
        self.reset()

    def forward(self, image, tstep):
        # Update membrane potential
        self.mempot.add_(image)
        mempot_ge_thres = self.mempot.ge(self.hyp.mempot_thres)
        self.mempot.sub_(mempot_ge_thres.type(torch.int), alpha=self.hyp.mempot_thres)

        # Calculate output spikes
        self.spikes[tstep] = mempot_ge_thres

    def reset(self):
        self.spikes = torch.zeros((self.tstep_cnt + 1, self.neu_in_cnt), dtype=torch.bool)
        self.mempot = torch.zeros(self.neu_in_cnt, dtype=torch.float32)

# Define network
class snn_model(nn.Module):
    def __init__(self, hyp):
        super().__init__()
        self.hyp = hyp

        # Layers
        neurons = hyp.neurons

        self.flat = nn.Flatten(0, -1)
        self.inp  = snn_input(neurons[0], hyp)
        self.lin1 = snn_linear(0, neurons[0], neurons[1], hyp)
        self.lin2 = snn_linear(1, neurons[1], neurons[2], hyp)
        self.lin3 = snn_linear(2, neurons[2], neurons[3], hyp)

    def forward(self, image):
        # Reset membrane potentials
        self.inp.reset()
        self.lin1.reset()
        self.lin2.reset()
        self.lin3.reset()

        # Process spikes
        image = self.flat(image)
        for tstep in range(self.hyp.tsteps):
            self.inp(image, tstep)
            self.lin1(self.inp.spikes[tstep], tstep, traces=self.training)
            self.lin2(self.lin1.spikes[tstep], tstep, traces=self.training)
            self.lin3(self.lin2.spikes[tstep], tstep, traces=self.training)

        return self.lin3.spikes

    def backward(self, label):
        # Desire of output layer
        error = torch.sum(self.lin3.spikes.type(torch.float32), dim=0).div(self.hyp.tsteps)
        error[label].neg_().add_(1)

        desire_0 = error.gt(self.hyp.desire_thres["out"])
        desire_1 = torch.zeros_like(desire_0)
        desire_1[label] = True
        self.lin3.desire = torch.stack((desire_0, desire_1), dim=1)

        # Backpropagate desire
        self.lin2.desire = self.lin3.backward(self.lin3.desire)
        self.lin1.desire = self.lin2.backward(self.lin2.desire)

# Define optimizer
class snn_optim(torch.optim.Optimizer):
    def __init__(self, model, hyp):
        self.model = model
        self.hyp   = hyp
        self.lr    = hyp.lr
        defaults = dict()

        super().__init__(model.parameters(), defaults)

    def step(self, closure=None):
        for layer in [self.model.lin1, self.model.lin2, self.model.lin3]:
            cond   = torch.logical_and(layer.spikes, layer.desire[:, 0].repeat(layer.tstep_cnt + 1, 1))
            sign   = layer.desire[:, 1].mul(2).sub(1).repeat(layer.tstep_cnt + 1, 1)
            update = layer.traces.repeat(layer.neu_post_cnt, 1, 1).permute(1, 0, 2)
            update.mul_(torch.mul(cond, sign).repeat(layer.neu_pre_cnt, 1, 1).permute(1, 2, 0))

            layer.weights.add_(torch.sum(update, dim=0), alpha=self.lr)

    def scheduler(self, epoch):
        # Exponential decay
        self.lr = self.hyp.lr * math.exp(-self.hyp.lr_decay * epoch)

# Define result computation
class snn_result:
    def __init__(self, hyp):
        self.neu_out = hyp.neurons[-1]
        self.results = None
        self.logfile = open(os.path.splitext(os.path.basename(__file__))[0] + ".log", "w", buffering=1)

    def register(self, spikes, label):
        spikes_max = torch.max(spikes)
        spikes_cnt = torch.bincount(spikes)
        if spikes[label] == spikes_max and spikes_cnt[spikes_max] == 1:
            self.results[0] += 1
        else:
            self.results[1] += 1

    def print(self, epoch, desc, length):
        accuracy = self.results[0] / self.results.sum() * 100
        print("Epoch {:2d} {:s}: {:.2f}%".format(epoch, desc, accuracy), file=self.logfile)

    def reset(self):
        self.results = torch.zeros(2, dtype=torch.int)

    def finish(self):
        self.logfile.close()


if __name__ == "__main__":
    # Parse hyper-parameters
    parser = argparse.ArgumentParser(description="Hyper-parameters for desire backpropagation")
    parser.add_argument("--tsteps", default=20, type=int, help="Number of time steps per image")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs")
    parser.add_argument("--lr", default=4.e-4, type=float, help="Learning rate for weight updates")
    parser.add_argument("--lr_decay", default=5e-2, type=float, help="Exponential decay for learning rate")
    parser.add_argument("--mempot_thres", default=1.0, type=float, help="Spike threshold for membrane potential")
    parser.add_argument("--mempot_decay", default=1, type=int, help="Decay rate for membrane potential and spike traces")
    parser.add_argument("--desire_thres", default=[0.05, 0.30], nargs=2, type=float, help="Hidden and output threshold for desire backpropagation")
    parser.add_argument("--shuffle_data", default=False, type=bool, help="Shuffle training dataset before every epoch")
    parser.add_argument("--random_seed", default=0, type=int, help="Random seed for weight initialization")

    hyper_pars = parser.parse_args()
    hyper_pars.neurons      = (784, 512, 256, 10)
    hyper_pars.lr           = hyper_pars.lr / hyper_pars.tsteps
    hyper_pars.desire_thres = {"hid": hyper_pars.desire_thres[0], "out": hyper_pars.desire_thres[1]}
    hyper_pars.gpu_ncpu     = torch.cuda.is_available()
    hyper_pars.device       = torch.device('cuda' if hyper_pars.gpu_ncpu else 'cpu')

    # Network configuration
    torch.set_default_tensor_type(torch.cuda.FloatTensor if hyper_pars.gpu_ncpu else torch.FloatTensor)
    torch.manual_seed(hyper_pars.random_seed)

    model = snn_model(hyper_pars)
    optim = snn_optim(model, hyper_pars)
    resul = snn_result(hyper_pars)

    # Iterate over MNIST images
    dataset_train = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
    dataset_test  = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())
    debug = True

    for epoch in range(hyper_pars.epochs):
        if debug: print(f"Epoch: {epoch}")

        # Training
        model.train()
        resul.reset()
        optim.scheduler(epoch)

        if hyper_pars.shuffle_data: dataorder_train = torch.randperm(len(dataset_train))
        else: dataorder_train = torch.arange(0, len(dataset_train), dtype=torch.int)

        for (image_cnt, image_idx) in enumerate(dataorder_train):
            image = dataset_train[image_idx][0].to(hyper_pars.device)
            label = torch.tensor(dataset_train[image_idx][1]).to(hyper_pars.device)
            if debug: print(f"Image {image_cnt}: {label}")

            spikes_out = torch.sum(model(image), dim=0)
            if debug: print(np.array(spikes_out.cpu()))
            resul.register(spikes_out, label)

            model.backward(label)
            optim.step()

        resul.print(epoch, "Training", len(dataset_train))

        # Inference
        model.eval()
        resul.reset()
        for (image_cnt, (image, label)) in enumerate(dataset_test):
            image, label = image.to(hyper_pars.device), torch.tensor(label).to(hyper_pars.device)
            if debug: print(f"Image {image_cnt}: {label}")

            spikes_out = torch.sum(model(image), dim=0)
            if debug: print(np.array(spikes_out.cpu()))
            resul.register(spikes_out, label)

        resul.print(epoch, "Test", len(dataset_test))

resul.finish()

