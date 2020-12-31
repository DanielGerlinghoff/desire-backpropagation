import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import itertools

# Define layers
class snn_linear(nn.Module):
    def __init__(self, idx, neu_in, neu_out, hyp):
        super().__init__()
        self.idx          = idx
        self.neu_pre_cnt  = neu_in
        self.neu_post_cnt = neu_out
        self.hyp          = hyp
        self.tstep_cnt    = hyp["tsteps"]
        self.decay        = (2 ** hyp["decay"] - 1) / 2 ** hyp["decay"]

        # Network parameters
        self.weights = nn.Parameter(torch.randn(neu_in, neu_out).mul(np.sqrt(2 / neu_in)))
        self.weights.requires_grad = False
        # TODO: Transpose weights to (neu_out, neu_in)

        self.reset()

    def forward(self, spikes_in, tstep, traces=False):
        # Update membrane potential
        self.mempot.addmv_(self.weights.t(), spikes_in.type(torch.float32))
        mempot_ge_thres = self.mempot.ge(self.hyp["mempot_thres"])
        self.mempot.sub_(mempot_ge_thres.type(torch.int), alpha=self.hyp["mempot_thres"])

        # Calculate output spikes
        self.spikes[tstep+1] = mempot_ge_thres

        # Decay membrane potential
        self.mempot.mul_(mempot_ge_thres.logical_not().mul(self.decay).add(mempot_ge_thres))

        # Generate traces
        if traces: self.gen_traces(spikes_in, tstep)

    def backward(self, desire_in):
        # Output error for desired and undesired neurons
        error_ndes = torch.sum(self.spikes.type(torch.float32), dim=0).div(self.tstep_cnt)
        error_des  = error_ndes.neg().add(1)
        error      = error_des.mul(desire_in[:, 1]) + error_ndes.mul(desire_in[:, 1].logical_not())

        # Sum weights and errors
        sign = desire_in[:, 1].mul(2).sub(1)
        sign.mul_(desire_in[:, 0])

        desire_sum = torch.sum(self.weights.mul(torch.mul(error, sign)), dim=1)

        # Desire of previous layer
        desire_out_0 = desire_sum.abs().ge(self.hyp["desire_thres"]["hidden"])
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
        self.tstep_cnt  = hyp["tsteps"]

        # Network parameters
        self.reset()

    def forward(self, image, tstep):
        # Update membrane potential
        self.mempot.add_(image)
        mempot_ge_thres = self.mempot.ge(self.hyp["mempot_thres"])
        self.mempot.sub_(mempot_ge_thres.type(torch.int), alpha=self.hyp["mempot_thres"])

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
        torch.manual_seed(0)
        neurons = hyp["neurons"]

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
        for tstep in range(self.hyp["tsteps"]):
            self.inp(image, tstep)
            self.lin1(self.inp.spikes[tstep], tstep, traces=self.training)
            self.lin2(self.lin1.spikes[tstep], tstep, traces=self.training)
            self.lin3(self.lin2.spikes[tstep], tstep, traces=self.training)

        return self.lin3.spikes

    def backward(self, label):
        # Desire of output layer
        error = torch.sum(self.lin3.spikes.type(torch.float32), dim=0).div(self.hyp["tsteps"])
        error[label].neg_().add_(1)

        desire_0 = error.gt(0)  # TODO: .gt(0), should be .ge(self.hyp["desire_thres"]["output"])
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
        defaults = dict()

        super().__init__(model.parameters(), defaults)

    def step(self, closure=None):
        for layer in [self.model.lin1, self.model.lin2, self.model.lin3]:
            cond   = torch.logical_and(layer.spikes, layer.desire[:, 0].repeat(layer.tstep_cnt + 1, 1))
            sign   = layer.desire[:, 1].mul(2).sub(1).repeat(layer.tstep_cnt + 1, 1)
            update = layer.traces.repeat(layer.neu_post_cnt, 1, 1).permute(1, 2, 0)
            update.mul_(torch.mul(cond, sign).repeat(layer.neu_pre_cnt, 1, 1).permute(1, 0, 2))

            layer.weights.add_(torch.sum(update, dim=0), alpha=self.hyp["learning_rate"])


if __name__ == "__main__":
    # Network configuration
    hyper_pars = dict()
    hyper_pars["neurons"]       = (784, 512, 256, 10)
    hyper_pars["tsteps"]        = 20
    hyper_pars["mempot_thres"]  = 1.0
    hyper_pars["learning_rate"] = 1.e-3 / hyper_pars["tsteps"]
    hyper_pars["decay"]         = 1
    hyper_pars["desire_thres"]  = {"hidden": 0.1, "output": 0.1}
    hyper_pars["gpu_ncpu"]      = torch.cuda.is_available()
    hyper_pars["device"]        = torch.device('cuda' if hyper_pars["gpu_ncpu"] else 'cpu')

    torch.set_default_tensor_type(torch.cuda.FloatTensor if hyper_pars["gpu_ncpu"] else torch.FloatTensor)

    model = snn_model(hyper_pars)
    optim = snn_optim(model, hyper_pars)

    # Iterate over MNIST images
    dataset_train = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
    dataset_test  = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())
    logfile = open(os.path.splitext(os.path.basename(__file__))[0] + ".log", "w", buffering=1)
    epoch_cnt = 1
    image_cnt = 2000
    debug     = True

    for epoch in range(epoch_cnt):
        if debug: print(f"Epoch: {epoch}")

        # Training
        model.train()
        for (image_idx, (image, label)) in enumerate(itertools.islice(dataset_train, image_cnt)):
            if debug: print(f"Image {image_idx}: {label}")
            image, label = image.to(hyper_pars["device"]), torch.tensor(label).to(hyper_pars["device"])

            spikes = model(image)
            model.backward(label)
            optim.step()
            if debug:
                print(np.array(torch.sum(spikes, dim=0).cpu()))

        # Export model
        if not os.path.exists(f"model"):
            os.makedirs(f"model")
        torch.save(model.state_dict(), f"model/model_{epoch}.pt")

        # Inference
        model.eval()
        results = torch.zeros((hyper_pars["neurons"][-1], hyper_pars["neurons"][-1]), dtype=torch.int)
        for (image_idx, (image, label)) in enumerate(itertools.islice(dataset_test, image_cnt)):
            if debug: print(f"Image {image_idx}: {label}")
            image, label = image.to(hyper_pars["device"]), torch.tensor(label).to(hyper_pars["device"])

            spikes_out = torch.sum(model(image), dim=0)
            if debug: print(np.array(spikes_out.cpu()))

            # Compare output with label
            spikes_max = torch.max(spikes_out)
            if spikes_out[label] == spikes_max:
                results[label][label] += 1
            else:
                results[label][torch.argmax(spikes_out)] += 1

        print(f"Epoch: {epoch}, Image: {image_idx}", file=logfile)
        print(np.array(results.cpu()), file=logfile)
        print("Accuracy: {:.2f}%".format(results.diag().sum() / image_cnt * 100), file=logfile)

logfile.close()

