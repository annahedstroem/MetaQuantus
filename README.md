<p align="center">
  <img width="350" src="https://raw.githubusercontent.com/annahedstroem/MetaQuantus/main/logo.png">
</p>
<!--<h1 align="center"><b>MetaQuantus</b></h1>-->
<h3 align="center"><b>Evaluate your XAI performance metrics</b></h3>
<p align="center">
  PyTorch and TensorFlow


[![Getting started!](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/understandable-machine-intelligence-lab/Quantus/blob/main/tutorials/Tutorial_ImageNet_Example_All_Metrics.ipynb)
[![Launch Tutorials](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/understandable-machine-intelligence-lab/Quantus/HEAD?labpath=tutorials)
[![Python package](https://github.com/understandable-machine-intelligence-lab/Quantus/actions/workflows/python-package.yml/badge.svg)](https://github.com/understandable-machine-intelligence-lab/Quantus/actions/workflows/python-package.yml)
![Python version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue.svg)
[![PyPI version](https://badge.fury.io/py/quantus.svg)](https://badge.fury.io/py/quantus)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Motivation
A ResNet9 model (He et al., 2016) is trained on CMNST dataset (Bykov et al., 2021) to classify digits
from 0 to 9. To understand the model’s prediction, we use several explanation methods including Saliency (Mørch
et al., 1995; Baehrens et al., 2010), Integrated Gradients (Sundararajan et al., 2017), and GradientShap (Lundberg &
Lee, 2017), which are distinguished by their respective colours. To evaluate the quality of the explanations, we apply
different estimators of faithfulness such as Faithfulness Correlation (FC) (Bhatt et al., 2020) and Pixel-Flipping (PF)
(Bach et al., 2015), which return a correlation coefficient and an AUC score, respectively. However, since the scores
vary depending on the estimator, both in range and direction, with lower or higher scores indicating more faithful
explanations, interpreting the resulting faithfulness scores is difficult

</p>
<p align="center">
  <img width="800" src="https://raw.githubusercontent.com/understandable-machine-intelligence-lab/Quantus/main/fig1.png">
</p>


With MetaQuantus, we provice a simple yet comprehensive framework called \texttt{MetaQuantus} whose primary purpose is to provide an objective, independent view of the estimator by evaluating it against two failure modes: resilience to noise and reactivity to adversary. 

## Citation

If you find this toolkit or its companion paper
[**Quantus: An Explainable AI Toolkit for Responsible Evaluation of Neural Network Explanations**](https://arxiv.org/abs/2202.06861)
interesting or useful in your research, use the following Bibtex annotation to cite us:

```bibtex
@article{hedstrom2023meta,
      title={Rethinking Performance Evaluation in Explainable AI with MetaQuantus}, 
      author={Anna Hedström and
              Philine Bommer and
              Kristoffer Wickström and
              Wojciech Samek and
              Sebastian Lapuschkin and
              Marina M.-C. Höhne},
      year={2023},
      eprint={2301.06861},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

When applying the individual metrics of Quantus, please make sure to also properly cite the work of the original authors (as linked above).

## Installation


If you already have [PyTorch](https://pytorch.org/) installed on your machine, 
the most light-weight version of MetaQuantus can be obtained from [PyPI](https://pypi.org/project/metaquantus/) as follows:

```setup
pip install metaquantus
```
Alternatively, you can simply add the desired deep learning framework (in brackets) to have the package installed together with MetaQuantus.
To install Quantus with PyTorch, please run:
```setup
pip install "metaquantus[torch]"
```

Alternatively, you can simply install MetaQuantus with [requirements.txt](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/main/requirements.txt).
Note that this installation requires that [PyTorch](https://pytorch.org/)are already installed on your machine.

```setup
pip install -r requirements.txt
```

<!-- pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -e --user # quantus!!!-->

## Getting started

To get started.....

## Reproduce experiments

To reproduce the experiments, first, you need to generate the results. This is done by running the python scripts as listed below. 
Ensure to have GPUs enabled at this stage as this will speed up computation considerably. Feel free to change the hyperparameters if you want to run similar experiments on other explanation methods,datasets or models. The results are then analysed in a separate notebook {LINK} where the visualisations are created.

Run benchmarking experiments.
```bash
python3 run_benchmarking.py --dataset=MNIST --fname=f --K=5 --iters=3
python3 run_benchmarking.py --dataset=fMNIST --fname=f --K=5 --iters=3
python3 run_benchmarking.py --dataset=cMNIST --fname=f --K=5 --iters=3
```

Run ranking example.
```bash
python3 run_ranking.py --dataset=cMNIST --fname=f --K=5 --iters=3 --category=Faithfulness
```

Run hp experiments.
```bash
python3 run_hp.py --dataset=MNIST --K=3 --iters=2
python3 run_hp.py --dataset=ImageNet --K=3 --iters=2
```

Run l dependency experiments. 
```bash
python3 run_l_dependency.py --dataset=MNIST --K=5 --iters=3 --reversed_order=False 
python3 run_l_dependency.py --dataset=fMNIST --K=5 --iters=3 --reversed_order=False 
python3 run_l_dependency.py --dataset=cMNIST --K=5 --iters=3 --reversed_order=False
```

Run sanity-checking exercise.
```bash
python3 run_hp.py --dataset=MNIST --K=3 --iters=2
python3 run_sanity_checks.py --dataset=ImageNet --K=3 --iters=2
```

Run test.
```bash
python3 run_test.py --dataset=ImageNet --K=3 --iters=2
```

