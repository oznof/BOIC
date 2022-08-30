# Bayesian Optimization with Informative Covariance

This repository will provide code to reproduce the experiments in the paper https://arxiv.org/abs/2208.02704
Additional results with trust regions can be found
[here (Section 2)](https://github.com/oznof/BOIC/blob/main/techrep.pdf)

## Informative Covariance

The project is concerned with more informative GP surrogates for Bayesian Optimization. In particular,
it proposes nonstationary covariance functions that
encode a priori preferences via 
[spatially-varying prior (co)variances](https://arxiv.org/pdf/2208.02704.pdf#figure.caption.4)
and promote heterogeneous local exploration via
[spatially-varying lengthscales](https://arxiv.org/pdf/2208.02704.pdf#figure.caption.3).
Both nonstationary effects are induced by a shaping function $\phi$ that ideally captures information about optima. 

Experiments aim to show that the proposed methodology with informative covariance functions remains useful even when
little is known in advance, as often is the case in black-box optimization. Hence, the focus on adaptive $\phi$.
An advantage of including information in the surrogate GP model itself is that $\phi$ and its parameters can be
fine-tuned by, e.g., empirical Bayes. 

Notably, this methodology can accommodate other nonnegative $\phi$, not just the 
[semiparametric form](https://arxiv.org/pdf/2208.02704.pdf#equation.3.11).
Still, this particular form is flexible enough that it can represent an arbitrary shape by appropriate choice of anchors,
weights, kernels and distance functions. There is enough flexibility to specify which choices should be learned
(and how), or kept fixed.

## Collapsed Expected Improvement

Interestingly, the problem shown in [Figure 1](https://arxiv.org/pdf/2208.02704.pdf#figure.caption.2) that is due to
overconfident stationary surrogates can also be solved by modifying the acquisition step that uses
Expected Improvement (EI). As described [here (Section 1)](https://github.com/oznof/BOIC/blob/main/techrep.pdf),
the proposed Collapsed EI (CEI) is based on the repeated application of the Laplace method with mode collapse.
The CEI algorithm leads to more informative acquisitions as it avoids locations with small predictive variances.
Naturally, being a complementary approach, it can be used in combination with informative surrogates,
and the information extracted during CEI can later be included in $\phi$.