import os
import numpy as np
from torchvision import datasets, transforms
import itertools

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

# Network parameters
weights = [np.empty(0)] * layer_cnt
mempot  = [np.empty(0)] * (layer_cnt + 1)
spikes  = [np.empty(0)] * (layer_cnt + 1)
traces  = [np.empty(0)] * layer_cnt
desire  = [np.empty(0)] * layer_cnt

np.random.seed(0)
for layer in range(layer_cnt):
    if train_ntest:
        # Kaiming initialization
        weights[layer] = np.random.randn(neurons[layer], neurons[layer+1]).astype(np.float32) * \
                         np.sqrt(2 / neurons[layer])
    else:
        weights[layer] = np.load(f"model/weights_{layer}.npy")

# Iterate over MNIST images
dataset = datasets.MNIST("data", download=True, train=train_ntest, transform=transforms.ToTensor())
logfile = open(os.path.splitext(os.path.basename(__file__))[0] + ("_train" if train_ntest else "_test") + ".log", "w")
for epoch in range(epoch_cnt):
    if debug: print(f"Epoch: {epoch}")

    for (image_idx, (image, label)) in enumerate(itertools.islice(dataset, image_cnt)):
        if debug: print(f"Image {image_idx}: {label}")
        if image_idx % debug_period == 0:
            results = np.zeros((neurons[-1], neurons[-1]), dtype=np.int)

        # Reset spikes and membrane potentials
        for layer in range(layer_cnt + 1):
            mempot[layer] = np.zeros(neurons[layer], dtype=np.float32)
            spikes[layer] = np.zeros((tstep_cnt + 1, neurons[layer]), dtype=np.bool)
            if layer < layer_cnt:
                traces[layer] = np.zeros((tstep_cnt + 1, neurons[layer]), dtype=np.float32)
                desire[layer] = np.zeros((neurons[layer+1], 2), dtype=np.bool)

        # Input image and forward pass
        image = np.array(image, dtype=np.float32).flatten()

        for tstep in range(tstep_cnt):
            # Generate input spike train from image
            for neu_in in range(neurons[0]):
                mempot[0][neu_in] += image[neu_in]
                if mempot[0][neu_in] >= mempot_thres:
                    spikes[0][tstep][neu_in] = True
                    mempot[0][neu_in] -= mempot_thres

            # Propagate spikes forward
            for layer in range(layer_cnt):
                for neu_post in range(neurons[layer+1]):
                    # Update membrane potential
                    for neu_pre in range(neurons[layer]):
                        if spikes[layer][tstep][neu_pre]:
                            mempot[layer+1][neu_post] += weights[layer][neu_pre][neu_post]

                    # Calculate output spikes and decay membrane potential
                    if mempot[layer+1][neu_post] >= mempot_thres:
                        spikes[layer+1][tstep+1][neu_post] = True
                        mempot[layer+1][neu_post] -= mempot_thres
                    else:
                        mempot_old = mempot[layer+1][neu_post]
                        mempot[layer+1][neu_post] = ((mempot_old * 2 ** decay) - mempot_old) / 2 ** decay

                # Update spike traces
                for neu_pre in range(neurons[layer]):
                    trace = traces[layer][tstep][neu_pre]
                    traces[layer][tstep+1][neu_pre] = ((trace * 2 ** decay) - trace) / 2 ** decay

                    if spikes[layer][tstep][neu_pre]:
                        traces[layer][tstep+1][neu_pre] += 1

        if debug:
            print(np.sum(spikes[-1], axis=0))

        # Compare output with label
        spikes_out = np.sum(spikes[-1], axis=0)
        spikes_max = np.max(spikes_out)
        if spikes_out[label] == spikes_max:
            results[label][label] += 1
        else:
            results[label][np.argmax(spikes_out)] += 1

        if image_idx % debug_period == debug_period - 1:
            print(f"Epoch: {epoch}, Image: {image_idx}", file=logfile)
            print(results, file=logfile)
            print("Accuracy: {:.2f}%".format(sum([results[idx, idx] for idx in range(neurons[-1])]) / debug_period * 100), file=logfile)
            logfile.flush()

        # Training portion
        if train_ntest:
            # Backpropagate desire
            for layer in [layer_cnt]:
                # Output layer
                for neu_pre in range(neurons[layer]):
                    if neu_pre == label:
                        desire[layer-1][neu_pre][0] = (1 - np.sum(spikes[layer][:,neu_pre]) / tstep_cnt) >= desire_thres["output"]
                        desire[layer-1][neu_pre][1] = True
                    else:
                        desire[layer-1][neu_pre][0] = (np.sum(spikes[layer][:,neu_pre]) / tstep_cnt) >= desire_thres["output"]
                        desire[layer-1][neu_pre][1] = False

            for layer in range(layer_cnt - 1, 0, -1):
                # Hidden layers
                for neu_pre in range(neurons[layer]):
                    desire_sum = 0
                    for neu_post in range(neurons[layer+1]):
                        if desire[layer][neu_post][0]:
                            if desire[layer][neu_post][1]:
                                target_mult = 1 - np.sum(spikes[layer+1][:,neu_post]) / tstep_cnt
                                desire_sum += weights[layer][neu_pre][neu_post] * target_mult
                            else:
                                target_mult = np.sum(spikes[layer+1][:,neu_post]) / tstep_cnt
                                desire_sum -= weights[layer][neu_pre][neu_post] * target_mult
                    desire[layer-1][neu_pre][0] = np.abs(desire_sum) >= desire_thres["hidden"]
                    desire[layer-1][neu_pre][1] = desire_sum > 0

            # Update weights
            for tstep in range(tstep_cnt):
                for layer in range(layer_cnt):
                    for neu_post in range(neurons[layer+1]):
                        if spikes[layer+1][tstep+1][neu_post] and desire[layer][neu_post][0]:
                            for neu_pre in range(neurons[layer]):
                                update = learning_rate * traces[layer][tstep+1][neu_pre]
                                weights[layer][neu_pre][neu_post] += (1 if desire[layer][neu_post][1] else -1) * update

    # Save network model
    if train_ntest:
        if not os.path.exists(f"model_{epoch}"):
            os.makedirs(f"model_{epoch}")
        for layer in range(layer_cnt):
            np.save(f"model_{epoch}/weights_{layer}.npy", weights[layer])

logfile.close()

