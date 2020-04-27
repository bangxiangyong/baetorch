# baetorch
Python library for Bayesian Autoencoders

## Features
- Quantify epistemic uncertainty using approximate Bayesian inference
  - MC-Dropout
  - Bayesian Ensembling (with Anchored priors)
  - Variational Inference (Bayes by Backprop)
- Options for specifying data likelihood p(X|theta) to Gaussian or Bernoulli
- Quantify (homo/heteroskedestic) aleatoric uncertainty using Gaussian Likelihood
- Automatic learning rate finder for Bayesian Autoencoders

## Evaluation and plots
### Samples from BAE-Ensemble (M=5), FashionMNIST vs MNIST
![ID](https://github.com/bangxiangyong/baetorch/blob/master/github_figures/ID-samples.png)
![OOD](https://github.com/bangxiangyong/baetorch/blob/master/github_figures/OOD-samples.png)

### Comparison of performance 
![ROC](https://github.com/bangxiangyong/baetorch/blob/master/github_figures/ROC.png)
![PRC](https://github.com/bangxiangyong/baetorch/blob/master/github_figures/PRC.png)

### Automatic learning rate finder (based on Leslie Smith's methods)
![Auto-learning-rate](https://github.com/bangxiangyong/baetorch/blob/master/github_figures/auto-learning-rate-finder.png)

## TODO:
- separate examples for VI, MCDropout, Ensemble
- show how to:
  - change fully-dense to convolutional autoencoders
  - use with/without learning rate finder
  - swap homoscedestic mode
  - run test suite

