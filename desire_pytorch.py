import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
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
        self.decay   = (2 ** hyp.decay - 1) / 2 ** hyp.decay

        # Netowrk parameters
        self.weights = nn.Parameter(torch.empty(chn_out, chn_in, dim_k, dim_k))
        nn.init.kaiming_normal_(self.weights)
        self.weights.requires_grad = False

        # Loss function
        self.loss = nn.L1Loss(reduction="none")

        self.reset()

    def forward(self, spikes_in, tstep, traces=False):
        # Update membrane potential
        self.mempot.add_(F.conv2d(spikes_in.unsqueeze(0).type(torch.float32), self.weights).squeeze(0))
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
        error = self.loss(desire_in[..., 1].type(torch.float32), spikes_sum.div(self.tsteps))

        # Sum weights and errors
        sign = desire_in[..., 1].mul(2).sub(1)
        sign.mul_(desire_in[..., 0])

        weights_flip = self.weights.flip((2, 3)).permute(1, 0, 2, 3)
        desire_sum   = F.conv2d(torch.mul(error, sign).unsqueeze(0), weights_flip, padding=self.dim_k-1).squeeze(0)

        # Desire of previous layer
        desire_out_0 = desire_sum.abs().ge(self.hyp.desire_thres["hid"])
        desire_out_1 = desire_sum.gt(0)
        desire_out   = torch.stack((desire_out_0, desire_out_1), dim=-1)

        return desire_out

    def gen_traces(self, spikes_in, tstep):
        self.traces[tstep+1] = self.traces[tstep].mul(self.decay)
        self.traces[tstep+1].add_(spikes_in.type(torch.int))

    def reset(self):
        self.spikes = torch.zeros((self.tsteps + 1, self.chn_out, self.dim_out, self.dim_out), dtype=torch.bool)
        self.mempot = torch.zeros((self.chn_out, self.dim_out, self.dim_out), dtype=torch.float32)
        self.traces = torch.zeros((self.tsteps + 1, self.chn_in, self.dim_in, self.dim_in), dtype=torch.float32)
        self.desire = torch.zeros((self.chn_out, self.dim_out, self.dim_out, 2), dtype=torch.bool)

class snn_linear(nn.Module):
    def __init__(self, neu_in, neu_out, hyp):
        super().__init__()
        self.neu_pre  = neu_in
        self.neu_post = neu_out
        self.hyp      = hyp
        self.tsteps   = hyp.tsteps
        self.decay    = (2 ** hyp.decay - 1) / 2 ** hyp.decay

        # Network parameters
        self.weights = nn.Parameter(torch.empty(neu_out, neu_in))
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
        error = self.loss(desire_in[:, 1].type(torch.float32), spikes_sum.div(self.tsteps))

        # Sum weights and errors
        sign = desire_in[:, 1].mul(2).sub(1)
        sign.mul_(desire_in[:, 0])

        desire_sum = torch.sum(self.weights.mul(torch.mul(error, sign).repeat(self.neu_pre, 1).t()), dim=0)

        # Desire of previous layer
        desire_out_0 = desire_sum.abs().ge(self.hyp.desire_thres["hid"])
        desire_out_1 = desire_sum.gt(0)
        desire_out   = torch.stack((desire_out_0, desire_out_1), dim=1)

        return desire_out

    def gen_traces(self, spikes_in, tstep):
        self.traces[tstep+1] = self.traces[tstep].mul(self.decay)
        self.traces[tstep+1].add_(spikes_in.type(torch.int))

    def reset(self):
        self.spikes = torch.zeros((self.tsteps + 1, self.neu_post), dtype=torch.bool)
        self.mempot = torch.zeros(self.neu_post, dtype=torch.float32)
        self.traces = torch.zeros((self.tsteps + 1, self.neu_pre), dtype=torch.float32)
        self.desire = torch.zeros((self.neu_post, 2), dtype=torch.bool)

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
        self.spikes[tstep] = mempot_ge_thres

    def reset(self):
        self.spikes = torch.zeros((self.tsteps + 1, self.chn_in, self.dim_in, self.dim_in), dtype=torch.bool)
        self.mempot = torch.zeros((self.chn_in, self.dim_in, self.dim_in), dtype=torch.float32)

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
        self.spikes[tstep] = spikes_in.view(-1)

    def backward(self, desire_in):
        desire_out = desire_in.view(self.chn_in, self.dim_in, self.dim_in, 2)
        return desire_out

    def reset(self):
        self.spikes = torch.zeros((self.tsteps + 1, self.neu_post), dtype=torch.bool)
        self.desire = torch.zeros((self.neu_post, 2), dtype=torch.bool)

# Define network
class snn_model(nn.Module):
    def __init__(self, hyp):
        super().__init__()
        self.hyp = hyp

        # Layers
        self.scale = nn.AvgPool2d(2)
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
        image = self.scale(image)
        for tstep in range(self.hyp.tsteps):
            for idx, layer in enumerate(self.layers):
                if type(layer) == snn_input:     layer(image, tstep)
                elif type(layer) == snn_conv:    layer(self.layers[idx-1].spikes[tstep], tstep, traces=self.training) 
                elif type(layer) == snn_flatten: layer(self.layers[idx-1].spikes[tstep], tstep)
                elif type(layer) == snn_linear:  layer(self.layers[idx-1].spikes[tstep], tstep, traces=self.training) 

        return self.layers[-1].spikes

    def backward(self, label):
        # Desire of output layer
        error = torch.sum(self.layers[-1].spikes.type(torch.float32), dim=0).div(self.hyp.tsteps)
        error[label].neg_().add_(1)

        desire_0 = error.gt(self.hyp.desire_thres["out"])
        desire_1 = torch.zeros_like(desire_0)
        desire_1[label] = True
        self.layers[-1].desire = torch.stack((desire_0, desire_1), dim=1)

        # Backpropagate desire
        for idx in range(len(self.layers) - 1, 1, -1):
            layer = self.layers[idx]
            self.layers[idx-1].desire = layer.backward(layer.desire)

# Define optimizer
class snn_optim(torch.optim.Optimizer):
    def __init__(self, model, hyp):
        self.model = model
        self.hyp   = hyp
        defaults = dict()

        super().__init__(model.parameters(), defaults)

    def step(self, closure=None):
        for layer in self.model.layers:
            if type(layer) == snn_conv:
                cond = torch.logical_and(layer.spikes, layer.desire[..., 0].expand_as(layer.spikes))
                sign = layer.desire[..., 1].mul(2).sub(1).expand_as(cond).type(torch.float32)
                for chn in range(layer.chn_in):
                    update = F.conv2d(layer.traces[:, chn, ...].unsqueeze(0), torch.mul(cond, sign).permute(1, 0, 2, 3)).squeeze(0)
                    layer.weights[:, chn, ...].add_(update, alpha=self.hyp.learning_rate)

            elif type(layer) == snn_linear:
                cond   = torch.logical_and(layer.spikes, layer.desire[:, 0].repeat(layer.tsteps + 1, 1))
                sign   = layer.desire[:, 1].mul(2).sub(1).repeat(layer.tsteps + 1, 1)
                update = layer.traces.repeat(layer.neu_post, 1, 1).permute(1, 0, 2)
                update.mul_(torch.mul(cond, sign).repeat(layer.neu_pre, 1, 1).permute(1, 2, 0))

                layer.weights.add_(torch.sum(update, dim=0), alpha=self.hyp.learning_rate)


# Define result computation
class snn_result:
    def __init__(self, hyp):
        self.neu_out = hyp.config[-1][1]
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
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs")
    parser.add_argument("--mempot_thres", default=1.0, type=float, help="Spike threshold for membrane potential")
    parser.add_argument("--learning_rate", default=1.e-4, type=float, help="Lerning rate for weight updates")
    parser.add_argument("--decay", default=1, type=int, help="Decay for membrane potential and spike traces")
    parser.add_argument("--desire_thres", default=[0.1, 0.0], nargs=2, type=float, help="Hidden and output threshold for desire backpropagation")
    parser.add_argument("--shuffle_data", default=True, type=bool, help="Shuffle training dataset before every epoch")
    parser.add_argument("--random_seed", default=0, type=int, help="Random seed for weight initialization")

    hyper_pars = parser.parse_args()
    hyper_pars.learning_rate = hyper_pars.learning_rate / hyper_pars.tsteps
    hyper_pars.desire_thres  = {"hid": hyper_pars.desire_thres[0], "out": hyper_pars.desire_thres[1]}
    hyper_pars.gpu_ncpu      = torch.cuda.is_available()
    hyper_pars.device        = torch.device('cuda' if hyper_pars.gpu_ncpu else 'cpu')

    # Network configuration
    hyper_pars.config = (
        ("I", 1, 14),  # Input: (input channels, input dimension)
        ("C", 10, 5),  # Convolution: (output channels, kernel size)
        ("C", 20, 5),
        ("F", ),       # Flatten
        ("L", 100),    # Linar: (output neurons)
        ("L", 10))

    torch.set_default_tensor_type(torch.cuda.FloatTensor if hyper_pars.gpu_ncpu else torch.FloatTensor)
    torch.manual_seed(hyper_pars.random_seed)

    model = snn_model(hyper_pars)
    optim = snn_optim(model, hyper_pars)
    resul = snn_result(hyper_pars)

    # Iterate over MNIST images
    dataset_train = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
    dataset_test  = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())
    debug = True

    if debug:
        debugfile = open(os.path.splitext(os.path.basename(__file__))[0] + ".dbg", "w", buffering=1)

    for epoch in range(hyper_pars.epochs):
        if debug: print(f"Epoch: {epoch}")

        # Training
        model.train()
        resul.reset()

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

        # Analyse model parameters
        if debug:
            print(f"Epoch: {epoch}", file=debugfile)
            for idx, layer in enumerate(model.layers):
                if type(layer) not in (snn_conv, snn_linear): continue

                print(f"Layer: {idx}", file=debugfile)
                if type(layer) == snn_conv:
                    mean = layer.weights.mean([2, 3])
                    std  = layer.weights.std([2, 3])
                elif type(layer) == snn_linear:
                    mean = layer.weights.mean(1)
                    std  = layer.weights.std(1)

                print("Mean:", file=debugfile)
                print(np.array(mean.cpu()), file=debugfile)
                print("Standard Deviation:", file=debugfile)
                print(np.array(std.cpu()), file=debugfile)

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
debugfile.close()

