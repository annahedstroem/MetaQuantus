<p align="center">
  <img width="750" src="https://github.com/annahedstroem/MetaQuantus/blob/main/logo.png?raw=true">
</p>
<!--<h1 align="center"><b>MetaQuantus</b></h1>-->
<h3 align="center"><b>A library to meta-evaluate XAI performance metrics</b></h3>
<p align="center">
  PyTorch

_MetaQuantus is currently under active development so carefully note the release version to ensure reproducibility of your work._

[![Getting started!](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/understandable-machine-intelligence-lab/Quantus/blob/main/tutorials/Tutorial_ImageNet_Example_All_Metrics.ipynb)
[![Launch Tutorials](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/understandable-machine-intelligence-lab/Quantus/HEAD?labpath=tutorials)
[![Python package](https://github.com/understandable-machine-intelligence-lab/Quantus/actions/workflows/python-package.yml/badge.svg)](https://github.com/understandable-machine-intelligence-lab/Quantus/actions/workflows/python-package.yml)
![Python version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue.svg)
[![PyPI version](https://badge.fury.io/py/quantus.svg)](https://badge.fury.io/py/quantus)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Motivation
An illustration of the Problem of Meta-Evaluation through three phases: (i) Modeling, (ii) Explaining and (iii) Evaluating. (i) A ResNet9 model \citep{he2015deep} is trained to classify digits from $0$ to $9$ on Customised-MNST dataset \citep{bykov2021noisegrad} (i.e., MNIST digits pasted on CIFAR-10 backgrounds). (ii) To understand the model's prediction, we use several explanation methods including \textit{Gradient} \citep{morch, baehrens}, \textit{Integrated Gradients} \citep{sundararajan2017axiomatic} and \textit{GradientShap} \citep{lundberg2017unified}, which are distinguished by their respective colours. (iii) To evaluate the quality of the explanations, we apply different estimators of faithfulness such as  \textit{Faithfulness Correlation} (FC) \citep{bhatt2020} and \textit{Pixel-Flipping} (PF) \citep{bach2015pixel}, which return a correlation coefficient and an AUC score, respectively. However, since the scores vary depending on the estimator, both in range and direction, with lower or higher scores indicating more faithful explanations, interpreting the resulting faithfulness scores remains difficult for the practitioner.

</p>
<p align="center">
  <img width="550" src="https://github.com/annahedstroem/MetaQuantus/blob/main/fig1-cmnist.png?raw=true">
</p>


With MetaQuantus, we address this problem by providing a simple yet comprehensive framework called \texttt{MetaQuantus} whose primary purpose is to provide an objective, independent view of the estimator by evaluating it against two failure modes: resilience to noise and reactivity to adversary. 

## Citation

If you find this toolkit or its companion paper
[**The Meta-Evaluation Problem in Explainable AI:
Rethinking Performance Estimation with MetaQuantus**](INSERT_PREPRINT_LINK)
interesting or useful in your research, use the following Bibtex annotation to cite us:

```bibtex
@article{hedstrom2023meta,
      title={The Meta-Evaluation Problem in Explainable AI: Rethinking Performance Evaluation in Explainable AI with MetaQuantus}, 
      author={anonymous},
      year={2023},
      eprint={INSERT},
      archivePrefix={INSERT},
      primaryClass={INSERT}
}
```

When applying the individual metrics of Quantus, please make sure to also properly cite the work of the original authors (as linked above).

## Installation

The most light-weight version of MetaQuantus can be obtained from [PyPI](https://pypi.org/project/metaquantus/) as follows:

```setup
pip install metaquantus
```

Alternatively, you can simply install MetaQuantus with [requirements.txt](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/main/requirements.txt).

```setup
pip install -r requirements.txt
```

Note that the installation requires that [PyTorch](https://pytorch.org/) is already installed on your machine.

### Package requirements

The package requirements are as follows:
```
python>=3.7.0
pytorch>=1.10.1
```

## Getting started

To get started, ......

