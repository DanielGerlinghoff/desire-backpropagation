import os
import numpy as np
from scipy import signal
from torchvision import datasets, transforms
import itertools

# Network configuration
neurons   = (
    (14, 0, 5, 1, (1, 10)),  # (dimension, padding, kernel, stride, (input channels, output channels))
    (10, 0, 5, 1, (10, 20)),
    720,
    100,
    10)
layer_cnt = len(neurons) - 1
tstep_cnt = 20
image_cnt = 60000
epoch_cnt = 1

train_ntest   = True
debug         = True
debug_period  = 5000
mempot_thres  = 1
learning_rate = 5e-4 / tstep_cnt
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
        if type(neurons[layer]) == tuple:
            weights[layer] = np.random.randn(neurons[layer][4][1], neurons[layer][4][0],
                             neurons[layer][2], neurons[layer][2]).astype(np.float32) * \
                             np.sqrt(2 / (neurons[layer][4][0] * neurons[layer][2] ** 2))
        elif type(neurons[layer]) == int:
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
            results = np.zeros(2, dtype=np.int)

        # Reset spikes and membrane potentials
        for layer in range(layer_cnt + 1):
            if type(neurons[layer]) == tuple:
                mempot[layer] = np.zeros((neurons[layer][4][0], neurons[layer][0], neurons[layer][0]), dtype=np.float32)
                spikes[layer] = np.zeros((tstep_cnt + 1, neurons[layer][4][0], neurons[layer][0], neurons[layer][0]), dtype=np.bool)
                if layer < layer_cnt:
                    traces[layer] = np.zeros((tstep_cnt + 1, neurons[layer][4][0], neurons[layer][0], neurons[layer][0]), dtype=np.float32)
                if layer > 0:
                    desire[layer-1] = np.zeros((neurons[layer][4][0], neurons[layer][0], neurons[layer][0], 2), dtype=np.bool)
            elif type(neurons[layer]) == int:
                mempot[layer] = np.zeros(neurons[layer], dtype=np.float32)
                spikes[layer] = np.zeros((tstep_cnt + 1, neurons[layer]), dtype=np.bool)
                if layer < layer_cnt:
                    traces[layer] = np.zeros((tstep_cnt + 1, neurons[layer]), dtype=np.float32)
                if layer > 0:
                    desire[layer-1] = np.zeros((neurons[layer], 2), dtype=np.bool)

        # Input image and forward pass
        image = np.array(image, dtype=np.float32)

        for tstep in range(tstep_cnt):
            # Generate input spike train from image
            for chn_in in range(neurons[0][4][0]):
                image_scale = signal.convolve2d(image[chn_in], np.full((2,2), 0.25, dtype=np.float32), mode="valid")[::2, ::2]
                for neu_row in range(neurons[0][0]):
                    for neu_col in range(neurons[0][0]):
                        mempot[0][chn_in, neu_row, neu_col] += image_scale[neu_row, neu_col]
                        if mempot[0][chn_in, neu_row, neu_col] >= mempot_thres:
                            spikes[0][tstep][chn_in, neu_row, neu_col] = True
                            mempot[0][chn_in, neu_row, neu_col] -= mempot_thres

            # Propagate spikes forward
            for layer in range(layer_cnt):
                if type(neurons[layer]) == tuple:
                    dim_out = int((neurons[layer][0] + 2 * neurons[layer][1] - neurons[layer][2]) / neurons[layer][3] + 1)
                    mempot_temp = mempot[layer+1] if type(neurons[layer+1]) == tuple else mempot[layer+1].reshape((neurons[layer][4][1], dim_out, dim_out))
                    spikes_temp = np.zeros((neurons[layer][4][1], dim_out, dim_out), dtype=np.bool)

                    for chn_out in range(neurons[layer][4][1]):
                        # Update membrane potential
                        for chn_in in range(neurons[layer][4][0]):
                            mempot_temp[chn_out] += signal.correlate2d(spikes[layer][tstep, chn_in], weights[layer][chn_out][chn_in], mode="valid")

                        # Calculate output spikes and decay membrane potential
                        for neu_row in range(dim_out):
                            for neu_col in range(dim_out):
                                if mempot_temp[chn_out, neu_row, neu_col] >= mempot_thres:
                                    spikes_temp[chn_out, neu_row, neu_col] = True
                                    mempot_temp[chn_out, neu_row, neu_col] -= mempot_thres
                                else:
                                    mempot_old = mempot_temp[chn_out, neu_row, neu_col]
                                    mempot_temp[chn_out, neu_row, neu_col] = ((mempot_old * 2 ** decay) - mempot_old) / 2 ** decay

                    if type(neurons[layer+1]) == tuple:
                        mempot[layer+1] = mempot_temp
                        spikes[layer+1][tstep+1] = spikes_temp
                    else:
                        mempot[layer+1] = mempot_temp.flatten()
                        spikes[layer+1][tstep+1] = spikes_temp.flatten()

                    # Update spike traces
                    for chn_in in range(neurons[layer][4][0]):
                        for neu_row in range(neurons[layer][0]):
                            for neu_col in range(neurons[layer][0]):
                                trace = traces[layer][tstep, chn_in, neu_row, neu_col]
                                traces[layer][tstep+1, chn_in, neu_row, neu_col] = ((trace * 2 ** decay) - trace) / 2 ** decay

                                if spikes[layer][tstep, chn_in, neu_row, neu_col]:
                                    traces[layer][tstep+1, chn_in, neu_row, neu_col] += 1

                elif type(neurons[layer]) == int:
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
        spikes_cnt = np.bincount(spikes_out)
        if spikes_out[label] == spikes_max and spikes_cnt[spikes_max] == 1:
            results[0] += 1
        else:
            results[1] += 1

        if image_idx % debug_period == debug_period - 1:
            accuracy = results[0] / results.sum() * 100
            print("Epoch {:2d} Image {:5d}: {:.2f}%".format(epoch, image_idx, accuracy), file=logfile)
            logfile.flush()

        # Training portion
        if train_ntest:
            # Backpropagate desire
            for layer in [layer_cnt]:
                # Output layer
                for neu_pre in range(neurons[layer]):
                    if neu_pre == label:
                        desire[layer-1][neu_pre][0] = (1 - np.sum(spikes[layer][:,neu_pre]) / tstep_cnt) > desire_thres["output"]
                        desire[layer-1][neu_pre][1] = True
                    else:
                        desire[layer-1][neu_pre][0] = (np.sum(spikes[layer][:,neu_pre]) / tstep_cnt) > desire_thres["output"]
                        desire[layer-1][neu_pre][1] = False

            for layer in range(layer_cnt - 1, 0, -1):
                # Hidden layers
                if type(neurons[layer]) == int:
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

                elif type(neurons[layer]) == tuple:
                    if type(neurons[layer+1]) == int:
                        dim_out = int((neurons[layer][0] + 2 * neurons[layer][1] - neurons[layer][2]) / neurons[layer][3] + 1)
                        desire_temp = desire[layer].reshape((neurons[layer][4][1], dim_out, dim_out, 2))
                        spikes_temp = spikes[layer+1].reshape((tstep_cnt + 1, neurons[layer][4][1], dim_out, dim_out))
                    else:
                        desire_temp = desire[layer]
                        spikes_temp = spikes[layer+1]

                    for chn_in in range(neurons[layer][4][0]):
                        desire_sum = np.zeros_like(desire[layer-1][chn_in,:,:,0], dtype=np.float32)
                        for chn_out in range(neurons[layer][4][1]):
                            sign = desire_temp[chn_out,:,:,0] * (desire_temp[chn_out,:,:,1] * 2 - 1)
                            spikes_sum = np.sum(spikes_temp[:,chn_out], axis=0)
                            error = np.abs(spikes_sum / tstep_cnt - desire_temp[chn_out,:,:,1].astype(np.float32))
                            desire_sum += signal.convolve2d(np.pad(sign * error, neurons[layer][2] - 1), weights[layer][chn_out, chn_in], mode="valid")
                        desire[layer-1][chn_in,:,:,0] = np.abs(desire_sum) >= desire_thres["hidden"]
                        desire[layer-1][chn_in,:,:,1] = desire_sum > 0

            # Update weights
            for tstep in range(tstep_cnt):
                for layer in range(layer_cnt):
                    if type(neurons[layer]) == tuple:
                        dim_out = int((neurons[layer][0] + 2 * neurons[layer][1] - neurons[layer][2]) / neurons[layer][3] + 1)
                        if type(neurons[layer+1]) == int:
                            desire_temp = desire[layer].reshape((neurons[layer][4][1], dim_out, dim_out, 2))
                            spikes_temp = spikes[layer+1].reshape((tstep_cnt + 1, neurons[layer][4][1], dim_out, dim_out))
                        else:
                            desire_temp = desire[layer]
                            spikes_temp = spikes[layer+1]

                        for chn_in in range(neurons[layer][4][0]):
                            for chn_out in range(neurons[layer][4][1]):
                                sign = desire_temp[chn_out,:,:,0] * (desire_temp[chn_out,:,:,1] * 2 - 1)
                                update = signal.correlate2d(traces[layer][tstep+1, chn_in], sign * spikes_temp[tstep+1, chn_out], mode="valid")
                                weights[layer][chn_out, chn_in] += learning_rate * update / dim_out ** 2

                    elif type(neurons[layer]) == int:
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

