# Desire Backpropagation
Training of multi-layer spiking neural networks using a combination of spike-timing-dependent plasiticiy (STDP) and backpropgatation of local errors.


## Usage
The only relevant file is `desire_pytorch.py`, which includes the implementation of the learning algorithm in PyTorch. `desire_backprop.py` is an early, slow version of the algorithm and `sgd_reference.py` serves as reference with ANN training performance.

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

Results will be written to `desire_pytorch.log`.

### Run the pre-trained models
```
python3 desire_pytorch.py --dataset mnist --model-path models/mnist.pt --eval
python3 desire_pytorch.py --dataset fashion-mnist --model-path models/fashion_mnist.pt --eval
```

## Citation
*D. Gerlinghoff, T. Luo, R. S. M. Goh and W. F. Wong, "Desire Backpropagation: A Lightweight Training Algorithm for Multi-Layer Spiking Neural Networks based on Spike-Timing-Dependent Plasticity," in Neurocomputing, doi: 10.1016/j.neucom.2023.126773.*

 [Access on *ScienceDirect*](https://www.sciencedirect.com/science/article/pii/S0925231223008962)
