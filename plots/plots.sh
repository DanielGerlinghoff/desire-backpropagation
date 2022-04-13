# Run training
pushd ..
python3 desire_pytorch.py --epochs 150 --dataset mnist --random-seed 0
popd

# Run plot processing
python3 error.py
