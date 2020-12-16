import numpy as np
from torchvision import datasets
import itertools

# Network configuration
neurons   = (784, 512, 256, 10)
layer_cnt = len(neurons) - 1
tstep_cnt = 20
image_cnt = 2000

train_ntest   = True
debug         = True
mempot_thres  = 1
learning_rate = 1.e-3 / tstep_cnt
decay         = 1

# Network parameters
weights = [np.empty(0)] * layer_cnt
mempot  = [np.empty(0)] * (layer_cnt + 1)
spikes  = [np.empty(0)] * (layer_cnt + 1)
traces  = [np.empty(0)] * layer_cnt
desire  = [np.empty(0)] * layer_cnt
spike_out_cnt = np.zeros(neurons[-1], dtype=np.int)
results = dict()

np.random.seed(0)
for layer in range(layer_cnt):
    if train_ntest:
        # Kaiming initialization
        weights[layer] = np.random.randn(neurons[layer], neurons[layer+1]) * np.sqrt(2 / neurons[layer])
    else:
        weights[layer] = np.load(f"model/weights_{layer}.npy")

# Iterate over MNIST images
dataset = datasets.MNIST("data/", download=True, train=True)
for (image_idx, (image, label)) in enumerate(itertools.islice(dataset, image_cnt)):
    if debug: print(f"Image {image_idx}: {label}")

    # Reset spikes and membrane potentials
    for layer in range(layer_cnt + 1):
        mempot[layer] = np.zeros(neurons[layer], dtype=np.float)
        spikes[layer] = np.zeros((tstep_cnt + 1, neurons[layer]), dtype=np.bool)
        if layer < layer_cnt:
            traces[layer] = np.zeros((tstep_cnt + 1, neurons[layer]), dtype=np.float)
            desire[layer] = np.zeros(neurons[layer+1], dtype=np.bool)
    
    # Input image and backpropagate desire
    image  = np.array(image).flatten().astype(np.float) / 255.

    desire[-1][label] = True
    for layer in range(layer_cnt - 1, 0, -1):
        for neu_pre in range(neurons[layer]):
            desire_sum = 0
            for neu_post in range(neurons[layer+1]):
                target_mult = 1
                if layer == layer_cnt - 1 and desire[-1][neu_post]:
                    target_mult *= neurons[-1]
                    target_mult *= tstep_cnt - spike_out_cnt[label]
                desire_sum -= (-1) ** desire[layer][neu_post] * weights[layer][neu_pre][neu_post] * target_mult
            desire[layer-1][neu_pre] = True if desire_sum > 0 else False

    # Process spikes and learn
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

            # Update weights
            if train_ntest:
                for neu_post in range(neurons[layer+1]):
                    if spikes[layer+1][tstep+1][neu_post]:
                        for neu_pre in range(neurons[layer]):
                            update = learning_rate * traces[layer][tstep+1][neu_pre]
                            weights[layer][neu_pre][neu_post] += (1 if desire[layer][neu_post] else -1) * update

    if debug:
        print(np.sum(spikes[-1], axis=0))

    spike_out_cnt[label] = np.sum(spikes[-1][:,label])

    # Compare output with label
    if not train_ntest:
        if label not in results.keys():
            results[label] = {True: 0, False: 0}

        spikes_out = np.sum(spikes[-1], axis=0)
        if np.argmax(spikes_out) == label:
            results[label][True] += 1
        else:
            results[label][False] += 1

# Save network model
if train_ntest:
    for layer in range(layer_cnt):
        np.save(f"model/weights_{layer}.npy", weights[layer])

pass
