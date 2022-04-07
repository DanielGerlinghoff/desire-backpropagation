# Desire Backpropagation
Training of multi-layer spiking neural networks using a combination of spike-timing-dependent plasiticiy (STDP) and backpropgatation of local errors.


## Usage
The only relevant file is desire\_pytorch.py, which includes the implementation of the learning algorithm in PyTorch. desire\_backprop.py is an early, slow version of the algorithm and sgd\_reference.py serves as reference with ANN training performance.

In addition to the usual training parameters, the follwing are added for desire backpropagation:

| Parameter    | Default          | Description                                                                     |
| ------------ | ---------------- | ------------------------------------------------------------------------------- |
| tsteps       | 20               | Length of spike trains used for each sample                                     |
| mempot-thres | 1.0              | Membrane potential that must be reached, before an output spike is fired        |
| mempot-decay | 2                | Decay of membrane potential and spike traces in the form `2**(-mempot_decay)`   |
| desire-thres | 0.20, 0.05, 0.30 | Tenarization thresholds for convolution, linear and output layers               |
| error-margin | 4                | Reduction of a spike count target for desired neuron to `tsteps-error_margin` |
| dropout      | 0.0, 0.3         | Dropout probability for input and hidden layers                                 |


### Reproduction of Paper Results
For MNIST dataset:
```
python3 desire_pytorch.py --dataset mnist --epochs 150
```

For Fashion MNIST dataset:
```
python3 desire_pytorch.py --dataset fashion-mnist --epochs 600 --dropout 0.05 0.40
```
