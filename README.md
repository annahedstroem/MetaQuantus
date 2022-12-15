# MetaQuantus
MetaQuantus is a layer on top of Quantus for analysis of metric performance

Installation.
```bash
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -e --user # quantus!!!
```

Run test.
```bash
python3 run_test.py --dataset=ImageNet --K=3 --iters=2
```

Run benchmarking experiments.
```bash
python3 run_benchmarking.py --dataset=MNIST --fname=f --K=5 --iters=3
python3 run_benchmarking.py --dataset=fMNIST --fname=f --K=5 --iters=3
python3 run_benchmarking.py --dataset=cMNIST --fname=f --K=5 --iters=3
```

```
python3 run_benchmarking.py --dataset=MNIST --fname=K10 --K=10 --iters=3
python3 run_benchmarking.py --dataset=fMNIST --fname=K10 --K=10 --iters=3
python3 run_benchmarking.py --dataset=cMNIST --fname=K10 --K=10 --iters=3
python3 run_benchmarking.py --dataset=MNIST --fname=I5 --K=5 --iters=5
python3 run_benchmarking.py --dataset=fMNIST --fname=I5 --K=5 --iters=5
python3 run_benchmarking.py --dataset=cMNIST --fname=I5 --K=5 --iters=5
```

Run L dependency experiments. 
```bash
python3 run_L_experiments.py --dataset=MNIST --K=5 --iters=3 --reversed_order=False 
python3 run_L_experiments.py --dataset=fMNIST --K=5 --iters=3 --reversed_order=False 
python3 run_L_experiments.py --dataset=cMNIST --K=5 --iters=3 --reversed_order=False
```

Run hp experiments.
```bash
python3 run_hp.py --dataset=MNIST --K=3 --iters=2
python3 run_hp.py --dataset=ImageNet --K=3 --iters=2
```

Run sanity-checking exercise.
```bash
python3 run_hp.py --dataset=MNIST --K=3 --iters=2
python3 run_sanity_checks.py --dataset=ImageNet --K=3 --iters=2
```


