# MetaQuantus
MetaQuantus is a layer on top of Quantus for analysis of metric performance

Installation.
```bash
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

pip 
Running experiments.
```bash
python3 run_benchmarking.py --dataset="fMNIST" --fname="" --K=5 --iters=3
```

python3 run_benchmarking.py --dataset="cMNIST" --fname="" --K=5 --iters=3

python3 run_hp.py --dataset=MNIST --K=3 --iters=2

