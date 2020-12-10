import numpy as np
from torchvision import datasets
import itertools

# Network configuration
neurons   = (784, 32, 32, 10)
layer_cnt = len(neurons) - 1
tstep_cnt = 10
image_cnt = 100

train_ntest   = True
debug         = True
mempot_thres  = 1
learning_rate = 0.001
decay         = 1

# Network parameters
weights = [np.empty(0)] * layer_cnt
mempot  = [np.empty(0)] * (layer_cnt + 1)
spikes  = [np.empty(0)] * (layer_cnt + 1)
traces  = [np.empty(0)] * layer_cnt
results = dict()

np.random.seed(0)
for layer in range(layer_cnt):
    if train_ntest:
        # Kaiming initialization
        weights[layer] = np.random.randn(neurons[layer], neurons[layer+1]) * np.sqrt(2 / neurons[layer])
    else:
        weights[layer] = np.load(f"model/weights_{layer}.npy")

# Function for learning algorithm
def update_weight(layer, tstep_post, neu_post, desire_post):
    for neu_pre in range(neurons[layer]):
        # Update weight
        update = learning_rate * traces[layer][tstep_post+1][neu_pre]
        update *= neurons[-1] if desire_post else 1
        update /= np.prod(neurons[layer+1:])
        weights[layer][neu_pre][neu_post] += update if desire_post else -update
        
        # Skip recursion at second layer
        if layer == 0:
            continue
        
        # Determine desire to spike
        desire_pre = not ((weights[layer][neu_pre][neu_post] > 0) ^ desire_post)

        # Call recursive
        update_weight(layer-1, tstep_post-1, neu_pre, desire_pre)

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
    
    # Input image and target
    image  = np.array(image).flatten().astype(np.float) / 255.
    target = np.zeros(neurons[-1], dtype=np.bool)
    target[label] = True

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

        # Backpropagate weight updates
        if train_ntest:
            for neu_out in range(neurons[-1]):
                if spikes[-1][tstep+1][neu_out] or neu_out == label:
                        desire_out = target[neu_out]
                        update_weight(layer_cnt - 1, tstep, neu_out, desire_out)

    # Print output spikes
    if debug:
        print(np.sum(spikes[-1], axis=0))

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
