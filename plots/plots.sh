# Run training
pushd ..
python3 desire_pytorch.py --epochs 5 --dataset mnist --no-shuffle-data --dropout 0.0 0.0
popd

# Run plot processing
python3 spikes.py
