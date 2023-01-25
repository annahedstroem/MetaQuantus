<p align="center">
  <img width="750" src="https://github.com/annahedstroem/MetaQuantus/blob/main/logo.png?raw=true">
</p>
<!--<h1 align="center"><b>MetaQuantus</b></h1>-->
<h3 align="center"><b>A library to meta-evaluate XAI performance metrics</b></h3>
<p align="center">
  PyTorch

_MetaQuantus is currently under active development so carefully note the release version to ensure reproducibility of your work._

[![Getting started!](https://colab.research.google.com/assets/colab-badge.svg)](anonymous)
[![Launch Tutorials](https://mybinder.org/badge_logo.svg)](anonymous)
![Python version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<!--[![Python package](https://github.com/understandable-machine-intelligence-lab/Quantus/actions/workflows/python-package.yml/badge.svg)](https://github.com/understandable-machine-intelligence-lab/Quantus/actions/workflows/python-package.yml)-->
<!--[![PyPI version](https://badge.fury.io/py/metaquantus.svg)](https://badge.fury.io/py/metaquantus)-->

## Motivation

This repository contains the code and experimental results for the paper "MetaQuantus: A Framework for Meta-Evaluation of Quality Estimators in Explainable AI"

### Problem
In Explainable AI, the problem of meta-evalaution, that is, the process of of evaluating the evaluation method, is crucial but often overlooked. This is particularly important when selecting and quantitatively comparing explanation methods for a given model, dataset, and task. However, the use of multiple metrics or evalaution techqniues can lead to conflicting results. For example, scores from different metrics vary, both in range and direction, with lower or higher scores indicating higher quality explanations, making it difficult for practitioners to interpret the scores and select the best explanation method. 

### Library

With MetaQuantus, we address this problem by providing a simple yet comprehensive framework whose primary purpose is to provide an objective, independent view of the estimator by evaluating it against two failure modes: resilience to noise and reactivity to adversary. In a similar way that software systems undergo vulnerability and penetration tests before deployment, this tool is designed to stress test the evalaution methods e.g., as provided by Quantus.

</p>
<p align="center">
  <img width="800" src="https://raw.githubusercontent.com/understandable-machine-intelligence-lab/Quantus/main/fig1.png">
</p>

MetaQuantus is the first open-sourced, general-purpose solution that support developers in XAI and ML with a theoretically-grounded, practical tool a meta-evaluate newly developed, or existing metrics. It provides a easy-to-use API that makes the selection of metrics easier, with a few lines of code, metrics can be evaluated and chosen in its unique explainability context. XAI explanation methods with minimal code.

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
quantus>=0.3.2
captum>=0.4.1
```


## Getting started

Please see [
Tutorial-Getting-Started-with-MetaQuantus.ipynb](anonymous) under tutorials/ folder to run code similar to this example. Note that [PyTorch](https://pytorch.org/) framework and the XAI evalaution library [Quantus](https://github.com/understandable-machine-intelligence-lab/Quantus) is needed to run MetaQuantus

  
## Reproduce the experiments

To reproduce the results of this paper, you will need to follow these steps:

1. Data Generation: Run the notebook [
Tutorial-Data-Generation-Experiments.ipynb](anonymous) to generate the necessary data for the analysis. This notebook will guide you through the process of downloading and preprocessing the data. Make sure to follow the instructions carefully and to have all the necessary packages installed.

2. Results Analysis: Once the data generation step is complete, run the [
Tutorial-Reproduce-Experiments.ipynb](anonymous) to analyse the results. Make sure to adjust local path so that approriate files can be retrieved including having all the necessary packages installed. Please note that the results may slightly vary depending on the random seed and other hyperparameters, but the overall trends and conclusions should remain the same.


