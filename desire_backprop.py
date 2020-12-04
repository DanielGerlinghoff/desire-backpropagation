import numpy as np

# Network configuration
neurons   = (3, 5, 5, 2)
layer_cnt = len(neurons) - 1
tstep_cnt = 8

mempot_thres  = 0.5
learning_rate = 0.1
decay         = 1

# Network parameters
weights = [np.empty(0)] * layer_cnt
mempot  = [np.empty(0)] * layer_cnt
spikes  = [np.empty(0)] * (layer_cnt + 1)
traces  = [np.empty(0)] * layer_cnt
desire  = [np.empty(0)] * layer_cnt

np.random.seed(0)
for layer in range(layer_cnt):
    weights[layer] = 2. * np.random.rand(neurons[layer], neurons[layer+1]) - 0.6

# Reset spikes and membrane potentials
for layer in range(layer_cnt):
    mempot[layer]   = np.zeros(neurons[layer+1], dtype=float)
    spikes[layer+1] = np.zeros((tstep_cnt + 1, neurons[layer+1]), dtype=np.bool)
    traces[layer]   = np.zeros((tstep_cnt + 1, neurons[layer]), dtype=float)
    desire[layer]   = np.zeros(neurons[layer+1], dtype=np.bool)

# Define input spikes and targets
spikes[0] = np.random.randint(0, 2, (tstep_cnt, neurons[0])).astype(bool)
desire[-1][np.random.randint(0, 2)] = True

# Function for learning algorithm
def update_weight(layer, tstep_post, neu_post, desire_post):
    for neu_pre in range(neurons[layer]):
        # Update weight
        update = learning_rate * traces[layer][tstep_post+1][neu_pre]
        weights[layer][neu_pre][neu_post] += update if desire_post else -update
        
        # Skip recursion at second layer
        if layer == 0:
            continue
        
        # Determine desire to spike
        desire_pre = not ((weights[layer][neu_pre][neu_post] > 0) ^ desire_post)

        # Call recursive
        update_weight(layer-1, tstep_post-1, neu_pre, desire_pre)

# Process spikes and learn
for tstep in range(tstep_cnt):
    # Propagate spikes forward
    for layer in range(layer_cnt):
        for neu_post in range(neurons[layer+1]):
            # Update membrane potential
            for neu_pre in range(neurons[layer]):
                if spikes[layer][tstep][neu_pre]:
                    mempot[layer][neu_post] += weights[layer][neu_pre][neu_post]

            # Calculate output spikes
            if mempot[layer][neu_post] > mempot_thres:
                mempot[layer][neu_post] = 0
                spikes[layer+1][tstep+1][neu_post] = True

        # Update spike traces
        for neu_pre in range(neurons[layer]):
            trace = traces[layer][tstep][neu_pre]
            traces[layer][tstep+1][neu_pre] = ((trace * 2 ** decay) - trace) / 2 ** decay

            if spikes[layer][tstep][neu_pre]:
                traces[layer][tstep+1][neu_pre] += 1

    # Backpropagate weight updates
    for neu_out in range(neurons[-1]):
        if spikes[-1][tstep+1][neu_out]:
                desire_out = desire[-1][neu_out]
                update_weight(layer_cnt - 1, tstep, neu_out, desire_out)

pass
