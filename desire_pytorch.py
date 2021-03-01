import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import math, copy
import argparse

# Define layers
class snn_conv(nn.Module):
    def __init__(self, chn_in, chn_out, dim_k, dim_in, hyp):
        super().__init__()
        self.chn_in  = chn_in
        self.chn_out = chn_out
        self.dim_k   = dim_k
        self.dim_in  = dim_in
        self.dim_out = dim_in - dim_k + 1
        self.hyp     = hyp
        self.tsteps  = hyp.tsteps
        self.decay   = (2 ** hyp.mempot_decay - 1) / 2 ** hyp.mempot_decay

        # Netowrk parameters
        self.weights = nn.Parameter(torch.empty(chn_out, chn_in, dim_k, dim_k))
        nn.init.kaiming_normal_(self.weights)
        self.weights.requires_grad = False

        # Loss function
        self.loss = nn.L1Loss(reduction="none")

        self.reset()

    def forward(self, spikes_in, tstep):
        # Update membrane potential
        self.mempot.add_(F.conv2d(spikes_in.type(torch.float32), self.weights))
        mempot_ge_thres = self.mempot.ge(self.hyp.mempot_thres)
        self.mempot.sub_(mempot_ge_thres.type(torch.int), alpha=self.hyp.mempot_thres)

        # Calculate output spikes
        self.spikes[:, tstep+1] = mempot_ge_thres

        # Decay membrane potential
        self.mempot.mul_(mempot_ge_thres.logical_not().mul(self.decay).add(mempot_ge_thres))

    def backward(self, desire_in):
        # Output error
        spikes_sum = torch.sum(self.spikes.type(torch.float32), dim=1)
        error = self.loss(desire_in[..., 1].type(torch.float32), spikes_sum.div(self.tsteps - self.hyp.error_margin))

        # Sum weights and errors
        sign = desire_in[..., 1].mul(2).sub(1)
        sign.mul_(desire_in[..., 0])

        weights_flip = self.weights.flip((2, 3)).permute(1, 0, 2, 3)
        desire_sum   = F.conv2d(torch.mul(error, sign), weights_flip, padding=self.dim_k-1)

        # Desire of previous layer
        desire_out_0 = desire_sum.abs().ge(self.hyp.desire_thres["conv"])
        desire_out_1 = desire_sum.gt(0)
        desire_out   = torch.stack((desire_out_0, desire_out_1), dim=-1)

        return desire_out

    def reset(self):
        self.spikes = torch.zeros((self.hyp.batch_size, self.tsteps + 1, self.chn_out, self.dim_out, self.dim_out), dtype=torch.bool)
        self.mempot = torch.zeros((self.hyp.batch_size, self.chn_out, self.dim_out, self.dim_out), dtype=torch.float32)
        self.desire = torch.zeros((self.hyp.batch_size, self.chn_out, self.dim_out, self.dim_out, 2), dtype=torch.bool)

class snn_linear(nn.Module):
    def __init__(self, neu_in, neu_out, hyp):
        super().__init__()
        self.neu_pre  = neu_in
        self.neu_post = neu_out
        self.hyp      = hyp
        self.tsteps   = hyp.tsteps
        self.decay    = (2 ** hyp.mempot_decay - 1) / 2 ** hyp.mempot_decay

        # Network parameters
        self.weights = nn.Parameter(torch.empty(neu_out, neu_in))
        nn.init.kaiming_normal_(self.weights)
        self.weights.requires_grad = False

        # Loss function
        self.loss = nn.L1Loss(reduction="none")

        self.reset()

    def forward(self, spikes_in, tstep, traces=False):
        # Update membrane potential
        self.mempot.add_(torch.matmul(
            self.weights.unsqueeze(0).expand(self.hyp.batch_size, -1, -1),
            spikes_in.type(torch.float32).unsqueeze(2)).squeeze(2))
        mempot_ge_thres = self.mempot.ge(self.hyp.mempot_thres)
        self.mempot.sub_(mempot_ge_thres.type(torch.int), alpha=self.hyp.mempot_thres)

        # Calculate output spikes
        self.spikes[:, tstep+1] = mempot_ge_thres

        # Decay membrane potential
        self.mempot.mul_(mempot_ge_thres.logical_not().mul(self.decay).add(mempot_ge_thres))

        # Generate traces
        if traces: self.gen_traces(spikes_in, tstep)

    def backward(self, desire_in):
        # Output error
        spikes_sum = torch.sum(self.spikes.type(torch.float32), dim=1)
        error = self.loss(desire_in[..., 1].type(torch.float32), spikes_sum.div(self.tsteps - self.hyp.error_margin))

        # Sum weights and errors
        sign = desire_in[..., 1].mul(2).sub(1)
        sign.mul_(desire_in[..., 0])

        error = torch.mul(error, sign).expand(self.neu_pre, -1, -1).permute(1, 2, 0)
        desire_sum = torch.sum(self.weights.expand(self.hyp.batch_size, -1, -1).mul(error), dim=1)

        # Desire of previous layer
        desire_out_0 = desire_sum.abs().ge(self.hyp.desire_thres["lin"])
        desire_out_1 = desire_sum.gt(0)
        desire_out   = torch.stack((desire_out_0, desire_out_1), dim=2)

        return desire_out

    def gen_traces(self, spikes_in, tstep):
        self.traces[:, tstep+1] = self.traces[:, tstep].mul(self.decay)
        self.traces[:, tstep+1].add_(spikes_in.type(torch.int))

    def reset(self):
        self.spikes = torch.zeros((self.hyp.batch_size, self.tsteps + 1, self.neu_post), dtype=torch.bool)
        self.mempot = torch.zeros((self.hyp.batch_size, self.neu_post), dtype=torch.float32)
        self.traces = torch.zeros((self.hyp.batch_size, self.tsteps + 1, self.neu_pre), dtype=torch.float32)
        self.desire = torch.zeros((self.hyp.batch_size, self.neu_post, 2), dtype=torch.bool)

class snn_input(nn.Module):
    def __init__(self, chn_in, dim_in, hyp):
        super().__init__()
        self.chn_in  = chn_in
        self.chn_out = chn_in
        self.dim_in  = dim_in
        self.dim_out = dim_in
        self.hyp     = hyp
        self.tsteps  = hyp.tsteps

        # Network parameters
        self.reset()

    def forward(self, image, tstep):
        # Update membrane potential
        self.mempot.add_(image)
        mempot_ge_thres = self.mempot.ge(self.hyp.mempot_thres)
        self.mempot.sub_(mempot_ge_thres.type(torch.int), alpha=self.hyp.mempot_thres)

        # Calculate output spikes
        self.spikes[:, tstep] = mempot_ge_thres

    def reset(self):
        self.spikes = torch.zeros((self.hyp.batch_size, self.tsteps + 1, self.chn_in, self.dim_in, self.dim_in), dtype=torch.bool)
        self.mempot = torch.zeros((self.hyp.batch_size, self.chn_in, self.dim_in, self.dim_in), dtype=torch.float32)

class snn_flatten(nn.Module):
    def __init__(self, chn_in, dim_in, hyp):
        super().__init__()
        self.chn_in   = chn_in
        self.dim_in   = dim_in
        self.neu_post = chn_in * dim_in ** 2
        self.hyp      = hyp
        self.tsteps   = hyp.tsteps

        # Network parameters
        self.reset()

    def forward(self, spikes_in, tstep):
        self.spikes[:, tstep] = spikes_in.view(self.hyp.batch_size, -1)

    def backward(self, desire_in):
        desire_out = desire_in.view(self.hyp.batch_size, self.chn_in, self.dim_in, self.dim_in, 2)
        return desire_out

    def reset(self):
        self.spikes = torch.zeros((self.hyp.batch_size, self.tsteps + 1, self.neu_post), dtype=torch.bool)
        self.desire = torch.zeros((self.hyp.batch_size, self.neu_post, 2), dtype=torch.bool)

# Define network
class snn_model(nn.Module):
    def __init__(self, hyp):
        super().__init__()
        self.hyp = hyp

        # Layers
        self.layers = nn.ModuleList()
        for idx, config in enumerate(hyp.config):
            if idx > 0: layer_prev = self.layers[idx-1]

            if config[0] == "I":   layer = snn_input(config[1], config[2], hyp)
            elif config[0] == "C": layer = snn_conv(layer_prev.chn_out, config[1], config[2], layer_prev.dim_out, hyp)
            elif config[0] == "F": layer = snn_flatten(layer_prev.chn_out, layer_prev.dim_out, hyp)
            elif config[0] == "L": layer = snn_linear(layer_prev.neu_post, config[1], hyp)
            self.layers.append(layer)

    def forward(self, image):
        # Reset membrane potentials
        for layer in self.layers:
            layer.reset()

        # Process spikes
        for tstep in range(self.hyp.tsteps):
            for idx, layer in enumerate(self.layers):
                spikes_prev = self.layers[idx-1].spikes[:, tstep]
                if type(layer) == snn_input:     layer(image, tstep)
                elif type(layer) == snn_conv:    layer(spikes_prev, tstep)
                elif type(layer) == snn_flatten: layer(spikes_prev, tstep)
                elif type(layer) == snn_linear:  layer(spikes_prev, tstep, traces=self.training)

        return self.layers[-1].spikes

    def backward(self, label):
        # Desire of output layer
        error = torch.sum(self.layers[-1].spikes.type(torch.float32), dim=1).div(self.hyp.tsteps - self.hyp.error_margin)
        for b in range(self.hyp.batch_size): error[b][label[b]].neg_().add_(1)

        desire_0 = error.gt(self.hyp.desire_thres["out"])
        desire_1 = torch.zeros_like(desire_0)
        desire_1[torch.arange(self.hyp.batch_size), label] = True
        self.layers[-1].desire = torch.stack((desire_0, desire_1), dim=2)

        # Backpropagate desire
        for idx in range(len(self.layers) - 1, 1, -1):
            layer = self.layers[idx]
            self.layers[idx-1].desire = layer.backward(layer.desire)

# Define optimizer
class snn_optim(torch.optim.Optimizer):
    def __init__(self, model, hyp):
        self.model = model
        self.hyp   = hyp
        self.lr    = copy.copy(hyp.lr)
        defaults = dict()

        super().__init__(model.parameters(), defaults)

    def step(self):
        for idx, layer in enumerate(self.model.layers):
            if type(layer) == snn_conv:
                spikes_in  = self.model.layers[idx-1].spikes.sum(dim=1).type(torch.float32)
                spikes_out = layer.spikes.sum(dim=1).type(torch.float32)
                error = torch.sub(spikes_out.div(layer.tsteps - self.hyp.error_margin), layer.desire[..., 1].type(torch.float32))
                cond  = layer.desire[..., 0].type(torch.float32)
                for chn in range(layer.chn_in):
                    update = F.conv3d(spikes_in[None, None, :, chn], torch.mul(error, cond).permute(1, 0, 2, 3).unsqueeze(1))
                    update.squeeze_().div_(layer.dim_out ** 2)
                    layer.weights[:, chn, ...].sub_(update, alpha=self.lr["conv"])

            elif type(layer) == snn_linear:
                cond   = torch.logical_and(layer.spikes, layer.desire[:, None, :, 0].expand(-1, layer.tsteps + 1, -1))
                sign   = layer.desire[:, None, :, 1].mul(2).sub(1).expand(-1, layer.tsteps + 1, -1)
                update = layer.traces.unsqueeze(2).repeat(1, 1, layer.neu_post, 1)
                update.mul_(torch.mul(cond, sign).unsqueeze(3).expand(-1, -1, -1, layer.neu_pre))
                layer.weights.add_(torch.sum(update, dim=(0, 1)), alpha=self.lr["lin"])

    def scheduler(self, epoch):
        # Exponential decay
        self.lr["conv"] = self.hyp.lr["conv"] * math.exp(-self.hyp.lr_decay * epoch)
        self.lr["lin"]  = self.hyp.lr["lin"] * math.exp(-self.hyp.lr_decay * epoch)

# Define result computation
class snn_result:
    def __init__(self, hyp):
        self.neu_out = hyp.config[-1][1]
        self.results = None
        self.logfile = open(os.path.splitext(os.path.basename(__file__))[0] + ".log", "w", buffering=1)
        self.reset()

    def register(self, spikes, label):
        for b in range(len(spikes)):
            spikes_max = torch.max(spikes[b])
            spikes_cnt = torch.bincount(spikes[b])
            if spikes[b][label[b]] == spikes_max and spikes_cnt[spikes_max] == 1:
                self.results[0] += 1
            else:
                self.results[1] += 1

    def print(self, epoch, desc):
        accuracy = self.results[0] / self.results.sum() * 100
        print("Epoch {:2d} {:s}: {:.2f}%".format(epoch, desc, accuracy), file=self.logfile)
        self.reset()

    def reset(self):
        self.results = torch.zeros(2, dtype=torch.int)

    def finish(self):
        self.logfile.close()


if __name__ == "__main__":
    # Parse hyper-parameters
    parser = argparse.ArgumentParser(description="Hyper-parameters for desire backpropagation")
    parser.add_argument("--tsteps", default=20, type=int, help="Number of time steps per image")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", default=4, type=int, help="Size of batches")
    parser.add_argument("--lr", default=[2e-5, 2e-4], nargs=2, type=float, help="Lerning rate for kernel and weight updates")
    parser.add_argument("--lr_decay", default=4e-2, type=float, help="Exponential decay for learning rate")
    parser.add_argument("--mempot_thres", default=1.0, type=float, help="Spike threshold for membrane potential")
    parser.add_argument("--mempot_decay", default=2, type=int, help="Decay rate for membrane potential and spike traces")
    parser.add_argument("--desire_thres", default=[0.20, 0.05, 0.30], nargs=3, type=float, help="Convolution, linear and output threshold for desire backpropagation")
    parser.add_argument("--error_margin", default=4, type=int, help="Reduction of spikes required to reach zero error")
    parser.add_argument("--shuffle_data", default=True, type=bool, help="Shuffle training dataset before every epoch")
    parser.add_argument("--random_seed", default=0, type=int, help="Random seed for weight initialization")
    parser.add_argument("--model_path", default="", type=str, help="Give path to load trained model")
    parser.add_argument("--no-gpu", action="store_false", help="Do not use GPU")

    hyper_pars = parser.parse_args()
    hyper_pars.lr           = dict(zip(("conv", "lin"), [lr / hyper_pars.tsteps for lr in hyper_pars.lr]))
    hyper_pars.desire_thres = dict(zip(("conv", "lin", "out"), hyper_pars.desire_thres))
    hyper_pars.gpu_ncpu     = torch.cuda.is_available() and hyper_pars.no_gpu
    hyper_pars.device       = torch.device('cuda' if hyper_pars.gpu_ncpu else 'cpu')
    debug = True

    # Network configuration
    hyper_pars.config = (
        ("I", 1, 28),  # Input: (input channels, input dimension)
        ("C", 10, 5),  # Convolution: (output channels, kernel size)
        ("C", 20, 5),
        ("F", ),       # Flatten
        ("L", 256),    # Linar: (output neurons)
        ("L", 10))

    torch.set_default_tensor_type(torch.cuda.FloatTensor if hyper_pars.gpu_ncpu else torch.FloatTensor)
    torch.manual_seed(hyper_pars.random_seed)

    model = snn_model(hyper_pars)
    optim = snn_optim(model, hyper_pars)
    resul = snn_result(hyper_pars)

    if hyper_pars.model_path:
        model.load_state_dict(torch.load(hyper_pars.model_path))

    # Iterate over MNIST images
    dataset_train  = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
    dataload_train = DataLoader(dataset_train, batch_size=hyper_pars.batch_size, shuffle=hyper_pars.shuffle_data, drop_last=True)
    dataset_test   = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())
    dataload_test  = DataLoader(dataset_test, batch_size=hyper_pars.batch_size, drop_last=True)

    for epoch in range(hyper_pars.epochs):
        if debug: print(f"Epoch: {epoch}")

        # Training
        model.train()
        optim.scheduler(epoch)

        for (batch_idx, batch) in enumerate(dataload_train):
            images, labels = batch[0].to(hyper_pars.device), batch[1].to(hyper_pars.device)
            if debug: print(f"Batch {batch_idx}:")

            spikes_out = torch.sum(model(images), dim=1)
            if debug:
                for b in range(hyper_pars.batch_size):
                    print(f" {labels[b]}:\t{np.array(spikes_out[b].cpu())}")
            resul.register(spikes_out, labels)

            model.backward(labels)
            optim.step()

        resul.print(epoch, "Training")

        # Inference
        model.eval()
        for (image_cnt, batch) in enumerate(dataload_test):
            image, label = batch[0].to(hyper_pars.device), batch[1].to(hyper_pars.device)
            if debug: print(f"Image {image_cnt}: {label[0]}")

            spikes_out = torch.sum(model(image), dim=1)
            if debug: print(np.array(spikes_out.squeeze(0).cpu()))
            resul.register(spikes_out, label)

        resul.print(epoch, "Test")

    # Export model
    torch.save(model.state_dict(), os.path.splitext(os.path.basename(__file__))[0] + ".pt")
    resul.finish()

