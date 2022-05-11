# Run training
echo "To check memory utilization run: pmap <pid> | tail -n 1"

pushd ..
python3 desire_pytorch.py
popd
