# Bayesian Optimization with Informative Covariance

This repository provides the code for the informative covariance in https://arxiv.org/abs/2208.02704, whose
implementation can be found under ```boic.botorch.gpytorch.kernels.informative.InformativeKernel```.

The Gaussian process ```boic.torch.gp.models.ExactGPModel``` is equipped with constant and quadratic mean functions as well as stationary (Matérn or RBF), cylindrical and informative covariance functions.
The methods ```set_train_data``` and ```update_tr_bounds``` handle greedy anchors and trust regions.
The model is trained with ```botorch.optim.fit.fit_gpytorch_scipy``` and ```botorch.optim.optimize.optimize_acqf``` optimizes the acquisition function ```botorch.acquisition.analytic.Expected Improvement```.

Settings can be found under ```boic.torch.gp.botorch.settings.BoTorchSettings```. The default data path is ```~/storage/data/boic/```.
See also the demo in ```experiments```.

Installation in a virtual environment with ```python 3.8.10```:
```
git clone https://github.com/oznof/BOIC.git
cd BOIC
pip3 install -e . -f https://download.pytorch.org/whl/torch_stable.html
```

## Informative Covariance

This project proposes more informative GP priors for Bayesian optimization. It introduces nonstationary covariance functions that encode a priori preferences through spatially-varying prior (co)variances and promote local exploration via spatially-varying lengthscales.
These nonstationary effects are induced by a shaping function that can capture information about optima. 

Experiments show that the proposed methodology can significantly outperform baselines in high-dimensional problems and under weak prior information, as often is the case in black-box optimization.
Including information directly in the surrogate GP model allows fine-tuning of the shaping function and its parameters by, e.g., empirical Bayes.

More generally, this methodology can accommodate other nonnegative shaping functions beyond the specific semiparametric form shown in Equation 11. Nevertheless, this particular form is versatile, being capable, in theory, of representing a desired shape by appropriate choice of anchors, weights, kernels and distance functions. This flexibility allows for specifying which choices should be learned (and how), or kept fixed.

## Deflated Acquisition Functions and Posterior over Promising Points

To address the issue depicted in Figure 1, caused by overly confident stationary surrogates, another possible solution is to modify the acquisition step.

[Deflated acquisition functions](https://oznof.github.io/mlprojects/), first proposed as Collapsed Expected Improvement (CEI, [technical report, August 2022](https://doi.org/10.5281/zenodo.24233796)), repeatedly apply the Laplace method to remove the modes of the acquisition function where the predictive variance is already small.
This leads to more informative acquisitions, as locations with small predictive variances are not queried.

The implementation is in `boic.botorch.deflated` (experimental, not part of the paper's experiments). `DeflatedAcquisitionFunction` wraps any acquisition function, and its name adds the prefix Deflated, as in `DeflatedExpectedImprovement`. It works on a log-density chosen by the kind of function: the function itself for log acquisition functions (LogEI, LogPI), its logarithm for positive ones (EI, PI), and the energy-based form for those that can be negative (UCB, posterior mean). A mode is removed by subtraction, as in the 2022 report, or by multiplication, in the spirit of deflation operators. The removed modes form a Gaussian mixture, returned by `search_distribution()`. Tests: `tests/test_deflated.py`.

This method may also be combined with informative priors. Since an approximate posterior over promising points is computed as a byproduct, this information may be used to update the shaping function that induces the nonstationary effects. In addition, the underlying idea offers a general strategy to estimate multimodal functions/distributions without the problems generally associated with 1-turn variational inference (mass-covering or mode-seeking).
