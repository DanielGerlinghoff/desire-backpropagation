import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import math, copy
import argparse
import time


# Define layers
class SnnConv(nn.Module):
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
        error = self.loss(desire_in[..., 1].type(torch.float32), spikes_sum.div(self.tsteps - self.hyp.error_margin))

        # Sum weights and errors
        sign = desire_in[..., 1].mul(2).sub(1)
        sign.mul_(desire_in[..., 0])

        weights_flip = self.weights.flip((2, 3)).permute(1, 0, 2, 3)
        desire_sum   = F.conv2d(torch.mul(error, sign).unsqueeze(0), weights_flip, padding=self.dim_k-1).squeeze(0)

        # Desire of previous layer
        desire_out_0 = desire_sum.abs().ge(self.hyp.desire_thres["conv"])
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


class SnnLinear(nn.Module):
    def __init__(self, neu_in, neu_out, dropout, hyp):
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

        # Loss and dropout
        self.loss = nn.L1Loss(reduction="none")
        self.dropout_dist  = torch.distributions.bernoulli.Bernoulli(1 - hyp.dropout["hid"]) if dropout else None
        self.dropout_scale = 1 / (1 - hyp.dropout["hid"]) if dropout else 1

        self.reset()

    def forward(self, spikes_in, tstep):
        # Decay membrane potential
        self.mempot.mul_(self.decay)

        # Dropout only during training
        dropout_mask  = self.dropout_mask if self.training else torch.ones_like(self.dropout_mask)
        dropout_scale = self.dropout_scale if self.training else 1

        # Update membrane potential
        self.mempot.addmv_(self.weights, spikes_in)
        spikes_out = self.mempot.ge(self.hyp.mempot_thres).logical_and(dropout_mask)
        self.mempot.sub_(spikes_out.type(torch.int), alpha=self.hyp.mempot_thres)

        # Calculate output spikes
        self.spikes[tstep+1] = spikes_out.mul(dropout_scale)

        # Generate traces
        if self.training: self.gen_traces(spikes_in.clamp(max=1.0), tstep)

    def backward(self, desire_in):
        # Output error
        spikes_sum = torch.sum(self.spikes, dim=0).div(self.dropout_scale)
        error = self.loss(desire_in[:, 1].type(torch.float32), spikes_sum.div(self.tsteps - self.hyp.error_margin))

        # Sum weights and errors
        sign = desire_in[:, 1].mul(2).sub(1)
        sign.mul_(torch.logical_and(desire_in[:, 0], self.dropout_mask))

        desire_sum = torch.sum(self.weights.mul(torch.mul(error, sign).repeat(self.neu_pre, 1).t()), dim=0)

        # Desire of previous layer
        desire_out_0 = desire_sum.abs().ge(self.hyp.desire_thres["lin"])
        desire_out_1 = desire_sum.gt(0)
        desire_out   = torch.stack((desire_out_0, desire_out_1), dim=1)

        return desire_out

    def gen_traces(self, spikes_in, tstep):
        self.traces[tstep+1] = self.traces[tstep].mul(self.decay)
        self.traces[tstep+1].add_(spikes_in)

    def reset(self):
        self.spikes = torch.zeros((self.tsteps + 1, self.neu_post), dtype=torch.float32)
        self.mempot = torch.zeros(self.neu_post, dtype=torch.float32)
        self.traces = torch.zeros((self.tsteps + 1, self.neu_pre), dtype=torch.float32)
        self.desire = torch.zeros((self.neu_post, 2), dtype=torch.bool)

        if self.dropout_dist is not None:
            self.dropout_mask = self.dropout_dist.sample((self.neu_post,)).type(torch.bool)
        else:
            self.dropout_mask = torch.ones(self.neu_post, dtype=torch.bool)


class SnnInput(nn.Module):
    def __init__(self, chn_in, dim_in, hyp):
        super().__init__()
        self.chn_in  = chn_in
        self.chn_out = chn_in
        self.dim_in  = dim_in
        self.dim_out = dim_in
        self.hyp     = hyp
        self.tsteps  = hyp.tsteps

        # Dropout
        self.dropout_dist  = torch.distributions.bernoulli.Bernoulli(1 - hyp.dropout["in"])
        self.dropout_scale = 1 / (1 - hyp.dropout["in"])

        self.reset()

    def forward(self, image, tstep):
        # Dropout only during training
        dropout_mask  = self.dropout_mask if self.training else torch.ones_like(self.dropout_mask)
        dropout_scale = self.dropout_scale if self.training else 1

        # Update membrane potential
        self.mempot.add_(image)
        spikes_out = self.mempot.ge(self.hyp.mempot_thres).logical_and(dropout_mask)
        self.mempot.sub_(spikes_out.type(torch.int), alpha=self.hyp.mempot_thres)

        # Calculate output spikes
        self.spikes[tstep] = spikes_out.mul(dropout_scale)

    def reset(self):
        self.spikes = torch.zeros((self.tsteps + 1, self.chn_in, self.dim_in, self.dim_in), dtype=torch.float32)
        self.mempot = torch.zeros((self.chn_in, self.dim_in, self.dim_in), dtype=torch.float32)

        self.dropout_mask = self.dropout_dist.sample((self.chn_in, self.dim_in, self.dim_in)).type(torch.bool)


class SnnFlatten(nn.Module):
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
        self.spikes = torch.zeros((self.tsteps + 1, self.neu_post), dtype=torch.float32)
        self.desire = torch.zeros((self.neu_post, 2), dtype=torch.bool)


# Define network
class SnnModel(nn.Module):
    def __init__(self, hyp):
        super().__init__()
        self.hyp = hyp

        # Layers
        self.layers = nn.ModuleList()
        for idx, config in enumerate(hyp.config):
            if idx > 0: layer_prev = self.layers[idx-1]
            layer_last = True if idx == len(hyp.config) - 1 else False

            if config[0] == "I":   layer = SnnInput(config[1], config[2], hyp)
            elif config[0] == "C": layer = SnnConv(layer_prev.chn_out, config[1], config[2], layer_prev.dim_out, hyp)
            elif config[0] == "F": layer = SnnFlatten(layer_prev.chn_out, layer_prev.dim_out, hyp)
            elif config[0] == "L": layer = SnnLinear(layer_prev.neu_post, config[1], not layer_last, hyp)
            self.layers.append(layer)

    def forward(self, image):
        # Reset membrane potentials
        for layer in self.layers:
            layer.reset()

        # Process spikes
        for tstep in range(self.hyp.tsteps):
            for idx, layer in enumerate(self.layers):
                if type(layer) == SnnInput:     layer(image, tstep)
                elif type(layer) == SnnConv:    layer(self.layers[idx-1].spikes[tstep], tstep, traces=self.training) 
                elif type(layer) == SnnFlatten: layer(self.layers[idx-1].spikes[tstep], tstep)
                elif type(layer) == SnnLinear:  layer(self.layers[idx-1].spikes[tstep], tstep)

        return self.layers[-1].spikes

    def backward(self, label):
        # Desire of output layer
        error = torch.sum(self.layers[-1].spikes.type(torch.float32), dim=0).div(self.hyp.tsteps - self.hyp.error_margin)
        error[label] = 1 - error[label]

        desire_0 = error.gt(self.hyp.desire_thres["out"])
        desire_1 = torch.zeros_like(desire_0)
        desire_1[label] = True
        self.layers[-1].desire = torch.stack((desire_0, desire_1), dim=1)

        # Backpropagate desire
        for idx in range(len(self.layers) - 1, 1, -1):
            layer = self.layers[idx]
            self.layers[idx-1].desire = layer.backward(layer.desire)


# Define optimizer
class SnnOptim(torch.optim.Optimizer):
    def __init__(self, model, hyp):
        self.model = model
        self.hyp   = hyp
        self.lr    = copy.copy(hyp.lr)
        defaults = dict()

        super().__init__(model.parameters(), defaults)

    def step(self):
        for idx, layer in enumerate(self.model.layers):
            if type(layer) == SnnConv:
                spikes_in  = self.model.layers[idx-1].spikes.sum(dim=0).type(torch.float32)
                spikes_out = layer.spikes.sum(dim=0).type(torch.float32)
                error = torch.sub(spikes_out.div(layer.tsteps - self.hyp.error_margin), layer.desire[..., 1].type(torch.float32))
                cond  = layer.desire[..., 0].type(torch.float32)
                for chn in range(layer.chn_in):
                    update = F.conv2d(spikes_in[chn, None, None], torch.mul(error, cond).unsqueeze(1)).squeeze(0)
                    layer.weights[:, chn, ...].sub_(update, alpha=(self.lr["conv"] / layer.dim_out ** 2))

            elif type(layer) == SnnLinear:
                cond   = layer.spikes.logical_and(torch.logical_and(layer.desire[:, 0], layer.dropout_mask).repeat(layer.tsteps + 1, 1))
                sign   = layer.desire[:, 1].mul(2).sub(1).repeat(layer.tsteps + 1, 1)
                update = layer.traces.repeat(layer.neu_post, 1, 1).permute(1, 0, 2)
                update.mul_(torch.mul(cond, sign).repeat(layer.neu_pre, 1, 1).permute(1, 2, 0))

                layer.weights.add_(torch.sum(update, dim=0), alpha=self.lr["lin"])

    def scheduler(self, epoch):
        # Exponential decay
        self.lr["conv"] = self.hyp.lr["conv"] * math.exp(-self.hyp.lr_decay * epoch)
        self.lr["lin"]  = self.hyp.lr["lin"] * math.exp(-self.hyp.lr_decay * epoch)


# Define result computation
class SnnResult:
    def __init__(self, hyp):
        self.neu_out = hyp.config[-1][1]
        self.results = None
        self.logfile = open(os.path.splitext(os.path.basename(__file__))[0] + ".log", "w", buffering=1)

        self.acc_best = 0
        self.new_best = False

    def register(self, spikes, label):
        spikes_max = torch.max(spikes)
        spikes_cnt = torch.bincount(spikes)
        if spikes[label] == spikes_max and spikes_cnt[spikes_max] == 1:
            self.results[0] += 1
        else:
            self.results[1] += 1

    def print(self, epoch, desc, length):
        accuracy = self.results[0] / self.results.sum() * 100
        print("Epoch {:3d} {:s}: {:.2f}%".format(epoch, desc, accuracy), file=self.logfile)

        if desc == "Test":
            if accuracy > self.acc_best:
                self.acc_best = accuracy
                self.new_best = True
                print(" " * 6 + f"New best: {accuracy:.2f}%", file=self.logfile)
            else:
                self.new_best = False

    def reset(self):
        self.results = torch.zeros(2, dtype=torch.int)

    def finish(self):
        self.logfile.close()


if __name__ == "__main__":
    # Parse hyper-parameters
    parser = argparse.ArgumentParser(description="Hyper-parameters for desire backpropagation")
    parser.add_argument("--tsteps", default=20, type=int, help="Number of time steps per image")
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs")
    parser.add_argument("--lr", default=[1e-6, 1e-5], nargs=2, type=float, help="Learning rate for kernel and weight updates")
    parser.add_argument("--lr-decay", default=4e-2, type=float, help="Exponential decay for learning rate")
    parser.add_argument("--dropout", default=[0.0, 0.3], nargs=2, type=float, help="Dropout probability for input and hidden layers")
    parser.add_argument("--mempot-thres", default=1.0, type=float, help="Spike threshold for membrane potential")
    parser.add_argument("--mempot-decay", default=2, type=int, help="Decay rate for membrane potential and spike traces")
    parser.add_argument("--desire-thres", default=[0.20, 0.05, 0.30], nargs=3, type=float, help="Convolution, linear and output threshold for desire backpropagation")
    parser.add_argument("--error-margin", default=4, type=int, help="Reduction of spikes required to reach zero error")
    parser.add_argument("--random-seed", default=None, type=int, help="Random seed for weight initialization")
    parser.add_argument("--model-path", default="", type=str, help="Give path to load trained model")
    parser.add_argument("--dataset", default="mnist", type=str, choices=["mnist", "fashion-mnist"], help="Dataset used for training")
    parser.add_argument("--no-shuffle-data", action="store_false", help="Shuffle training dataset before every epoch")
    parser.add_argument("--no-gpu", action="store_false", help="Do not use GPU")
    parser.add_argument("--debug", action="store_true", help="Print output for debugging")

    hyper_pars = parser.parse_args()
    hyper_pars.lr           = dict(zip(("conv", "lin"), hyper_pars.lr))
    hyper_pars.desire_thres = dict(zip(("conv", "lin", "out"), hyper_pars.desire_thres))
    hyper_pars.dropout      = dict(zip(("in", "hid"), hyper_pars.dropout))
    hyper_pars.gpu_ncpu     = torch.cuda.is_available() and hyper_pars.no_gpu
    hyper_pars.device       = torch.device('cuda' if hyper_pars.gpu_ncpu else 'cpu')
    hyper_pars.shuffle_data = hyper_pars.no_shuffle_data

    # Load dataset
    if hyper_pars.dataset == "mnist":
        dataset_train = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
        dataset_test  = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())
    elif hyper_pars.dataset == "fashion-mnist":
        dataset_train = datasets.FashionMNIST("data", train=True, download=True, transform=transforms.ToTensor())
        dataset_test  = datasets.FashionMNIST("data", train=False, download=True, transform=transforms.ToTensor())

    # Network configuration
    # Input layer:   ("I", input channels, input dimensions)
    # Flatten layer: ("F", )
    # Linear layer:  ("L", output neurons)
    for n in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]:
        if hyper_pars.dataset == "mnist":
            hyper_pars.config = (
                ("I", 1, 28),
                ("F", ),
                ("L", n),
                ("L", 10))
        elif hyper_pars.dataset == "fashion-mnist":
            hyper_pars.config = (
                ("I", 1, 28),
                ("F", ),
                ("L", 1000),
                ("L", 100),
                ("L", 10))

        torch.set_default_tensor_type(torch.cuda.FloatTensor if hyper_pars.gpu_ncpu else torch.FloatTensor)
        if hyper_pars.random_seed is not None:
            torch.manual_seed(hyper_pars.random_seed)

        model = SnnModel(hyper_pars)
        optim = SnnOptim(model, hyper_pars)
        resul = SnnResult(hyper_pars)

        # Iterate over images
        for epoch in range(hyper_pars.epochs):
            if hyper_pars.debug: print(f"Epoch: {epoch}")

            # Load model
            if epoch:
                model.load_state_dict(torch.load(os.path.splitext(os.path.basename(__file__))[0] + ".pt"))
            elif hyper_pars.model_path:
                model.load_state_dict(torch.load(hyper_pars.model_path))

            # Training
            model.train()
            resul.reset()
            optim.scheduler(epoch)

            if hyper_pars.shuffle_data: dataorder_train = torch.randperm(len(dataset_train))
            else: dataorder_train = torch.arange(0, len(dataset_train), dtype=torch.int)

            time_fw = list()
            time_bw = list()
            time_step = list()
            for (image_cnt, image_idx) in enumerate(dataorder_train):
                if image_cnt == 2000: break
                image = dataset_train[image_idx][0].to(hyper_pars.device)
                label = torch.tensor(dataset_train[image_idx][1]).to(hyper_pars.device)
                if hyper_pars.debug: print(f"Image {image_cnt}: {label}")

                start_fw = time.time()
                spikes_out = torch.sum(model(image), dim=0).type(torch.int)
                time_fw.append(time.time() - start_fw)

                if hyper_pars.debug: print(np.array(spikes_out.cpu()))
                resul.register(spikes_out, label)

                start_bw = time.time()
                model.backward(label)
                time_bw.append(time.time() - start_bw)

                start_step = time.time()
                optim.step()
                time_step.append(time.time() - start_step)

            print(f"{n},", end="")
            for t in [time_fw, time_bw, time_step]:
                print(f"{min(t)},{sum(t)/len(t)},{max(t)},", end="")
            print("")
        continue

        resul.print(epoch, "Training", len(dataset_train))

        # Inference
        model.eval()
        resul.reset()
        for (image_cnt, (image, label)) in enumerate(dataset_test):
            image, label = image.to(hyper_pars.device), torch.tensor(label).to(hyper_pars.device)
            if hyper_pars.debug: print(f"Image {image_cnt}: {label}")

            spikes_out = torch.sum(model(image), dim=0).type(torch.int)
            if hyper_pars.debug: print(np.array(spikes_out.cpu()))
            resul.register(spikes_out, label)

        resul.print(epoch, "Test", len(dataset_test))

        # Export model
        torch.save(model.state_dict(), os.path.splitext(os.path.basename(__file__))[0] + ".pt")
        if resul.new_best:
            state = {"state": model.state_dict(), "pars": hyper_pars}
            torch.save(state, os.path.splitext(os.path.basename(__file__))[0] + f"_{epoch:03d}.pt")

    resul.finish()
