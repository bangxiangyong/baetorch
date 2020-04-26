# baetorch
Python library for Bayesian Autoencoders

## Evaluation and plots
### Samples from BAE-Ensemble (M=5), FashionMNIST vs MNIST
Predictions on In-distribution test data (FashionMNIST)
![Predictions on In-distribution test data (FashionMNIST)](https://github.com/bangxiangyong/baetorch/blob/master/github_figures/ID-samples.png)

Predictions on Out of distribution data (MNIST)
![Predictions on Out of distribution data (MNIST)](https://github.com/bangxiangyong/baetorch/blob/master/github_figures/OOD-samples.png)

### Comparison of performance based on epistemic uncertainty
AUROC curve
![ROC](https://github.com/bangxiangyong/baetorch/blob/master/github_figures/ROC.png)

AUPRC curve
![PRC](https://github.com/bangxiangyong/baetorch/blob/master/github_figures/PRC.png)

### Automatic learning rate finder (based on Leslie Smith)
Automatic learning rate finder
![Auto-learning-rate](https://github.com/bangxiangyong/baetorch/blob/master/github_figures/auto-learning-rate-finder.png)

## Features
- Quantify epistemic uncertainty using approximate Bayesian inference
  - MC-Dropout
  - Bayesian Ensembling
  - Variational Inference (Bayes by Backprop)
- Quantify (homo-hetero) aleatoric uncertainty using Gaussian Likelihood
- Automatic learning rate finder for Bayesian Autoencoders

TODO:
- separate examples for VI, MCDropout, Ensemble
- show how to:
  - change fully-dense to convolutional autoencoders
  - use with/without learning rate finder
  - swap homoscedestic mode
  - run test suite

Simplicity is a virtue ! 
