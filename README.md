# Certified Neural Quantile Regression

## Architecture

### Lipschitz neural network in JAX

Following the design principles of the library [Deel.Lip](https://github.com/deel-ai/deel-lip) distributed under MIT licence, we re-implement Lipschitz layers in Jax/Flax.

### Convex layers

Following the design principles of the library [OTT-jax](https://github.com/ott-jax/ott) distributed under Apache V.2.0 license, we re-implement input convex layers in Jax/Flax.

## Certified Quantile Regression

We propose a *parametric method* to estimate high dimensional confidence intervals, using Center Outward Distribution for quantile regression. [1]

[1] del Barrio, E., Sanz, A.G. and Hallin, M., 2022.
Nonparametric Multiple-Output Center-Outward Quantile Regression.
arXiv preprint [arXiv:2204.11756](https://arxiv.org/abs/2204.11756)
