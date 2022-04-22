# Run training
pushd ..
export PYTHONPATH=$PWD
python3 desire_pytorch.py --epochs 1 --dataset mnist --no-shuffle-data --dropout 0.0 0.0
popd

# Run plot processing
python3 desire.py
