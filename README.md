# Bayesian Optimization with Informative Covariance

This repository will provide the code for the informative covariance implemented in https://arxiv.org/abs/2208.02704

## Informative Covariance

This project proposes more informative GP priors for Bayesian optimization. It introduces nonstationary covariance functions that encode a priori preferences through spatially-varying prior (co)variances and promote local exploration via spatially-varying lengthscales.
These nonstationary effects are induced by a shaping function that can capture information about optima. 

Experiments show that the proposed methodology can significantly outperform baselines in high-dimensional problems and under weak prior information, as often is the case in black-box optimization.
Including information directly in the surrogate GP model allows fine-tuning of the shaping function and its parameters by, e.g., empirical Bayes.

More generally, this methodology can accommodate other nonnegative shaping functions beyond the specific semiparametric form shown in Equation 11. Nevertheless, this particular form is versatile, being capable, in theory, of representing a desired shape by appropriate choice of anchors, weights, kernels and distance functions. This flexibility allows for specifying which choices should be learned (and how), or kept fixed.

## Collapsed Expected Improvement and Posterior over Promising Points

To address the issue depicted in Figure 1, caused by overly confident stationary surrogates, another possible solution is to modify the acquisition step that uses Expected Improvement (EI).

The proposed [Collapsed EI (CEI)](https://aflme.github.io/mlprojects/#mode-collapsed-acquisition-functions) is based on the repeated application of the Laplace method with mode collapse.
This effectively leads to more informative acquisitions as locations with small predictive variances are not queried.

This method may also be combined with informative priors. Since an approximate posterior over promising points is computed as a byproduct, this information may be used to update the shaping function that induces the nonstationary effects. In addition, the underlying idea offers a general strategy to estimate multimodal functions/distributions without the problems generally associated with 1-turn variational inference (mass-covering or mode-seeking).
