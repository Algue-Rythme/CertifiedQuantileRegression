# This file is part of CNQR which is released under the Apache License 2.0.
# See file LICENSE in the root directory or https://opensource.org/licenses/Apache-2.0 for full license details.

"""Implement of Lipschitz neural network layer in Jax.

Freely inspired by Deel.Lip package: https://github.com/deel-ai/deel-lip distributed under MIT licence.
"""


from typing import Any, Callable, Iterable, List
from typing import Optional, Sequence, Tuple, Union

import jax
from jax import lax
import jax.numpy as jnp
import jax.nn.initializers as initializers
import flax
import flax.linen as nn

from cnqr.loop import while_loop


MAXITER_SPECTRAL_DEFAULT = 3
MAXITER_BJORCK_DEFAULT = 15


def l2_normalize(vec, epsilon=1e-5):
    norm = jnp.linalg.norm(vec) + epsilon
    return vec / norm


def power_iteration(kernel, u_init, maxiter_spectral,
                    unroll=True, jit=True):
    """Power iteration for finding the largest eigenvalue and eigenvector.

    See http://blog.mrtz.org/2013/12/04/power-method.html
    for fun facts about power iteration.

    Don't forget to celebrate its 100th birthday in 2029.
    
    Args:
        kernel: array of shape (A, B)
        u_init: initial vector of shape (A)
        maxiter_spectral: maximum number of iterations
        unroll: whether to unroll the loop (default: False)
        jit: whether to use jit (default: True)

    Returns:
        right eigenvector, left eigenvector
    """
    
    def body_fun(t):
        u, v = t
        v = l2_normalize(jnp.matmul(kernel, u))
        u = l2_normalize(jnp.matmul(v, kernel))
        return u, v

    v_init = l2_normalize(jnp.matmul(kernel, u_init))
    u, v = while_loop(cond_fun=(lambda _: True),
                      body_fun=body_fun,
                      init_val=(u_init, v_init),
                      maxiter=maxiter_spectral,
                      unroll=unroll, jit=jit)

    return u, v


def spectral_normalization(kernel, u, maxiter_spectral,
                           jit=True, unroll=True):
    """Spectral normalization of kernel.
    
    Args:
        kernel: array of shape (A, B)
        u: initial vector of shape (A)
        maxiter_spectral: maximum number of iterations
    
    Returns:
        normalized kernel, normalized u, spectral norm
    """
    u, v = power_iteration(kernel, u, maxiter_spectral, jit=jit, unroll=unroll)
    sigma = jnp.dot(v, jnp.dot(kernel, u))
    unit_spectral_norm_kernel = kernel / sigma
    return unit_spectral_norm_kernel, u, sigma


def bjorck_algorithm(kernel, maxiter_bjorck,
                     beta=0.5, order=1,
                     unroll=True, jit=True):
    """Bjorck algorithm [1] for finding the orthogonal kernel.

    kernel must be normalized.

    Original Bjorck's algorithm corresponds to beta=0.5 :
        For order = infty, convergence is only guaranteed if the spectral norm of the kernel is 1.
        For order 1, convergence is only guaranteed if the spectral norm of the kernel is smaller than sqrt(3)

    Other values of beta correspond to Weight Decay obtained by orthogonal regularization
    with beta*\|K^TK-I\|_2^2 like in Parseval's network [2].

    [1] Björck, Å. and Bowie, C., 1971.
    An iterative algorithm for computing the best estimate of an orthogonal matrix.
    SIAM Journal on Numerical Analysis, 8(2), pp.358-364.
    https://epubs.siam.org/doi/epdf/10.1137/0708036

    [2] Cisse, M., Bojanowski, P., Grave, E., Dauphin, Y. and Usunier, N., 2017, July.
    Parseval networks: Improving robustness to adversarial examples.
    In International Conference on Machine Learning (pp. 854-863). PMLR.
    
    Args:
        kernel: array of shape (A, B)
        maxiter_bjorck: maximum number of iterations
        beta: beta parameter for the algorithm (default: 0.5)
        order: order of Bjorck's algorithm. (default: 1). Higher values unsupported.
        unroll: whether to unroll the loop (default: False)
        jit: whether to use jit (default: True)
        
    Returns:
        orthogonal kernel
    """
    assert order == 1, "Higher order Bjorck's algorithm is not supported."
    
    def body_fun(kernel):
        _32_W = (1 + beta) * kernel
        _12_WWW = beta * kernel @ kernel.T @ kernel
        return _32_W - _12_WWW

    orthogonal_kernel = while_loop(cond_fun=lambda _: True,
                                   body_fun=body_fun,
                                   init_val=kernel,
                                   maxiter=maxiter_bjorck,
                                   unroll=unroll, jit=jit)
    
    return orthogonal_kernel


def kernel_orthogonalization(kernel,
                             u=None,
                             maxiter_spectral=MAXITER_SPECTRAL_DEFAULT,
                             maxiter_bjorck=MAXITER_BJORCK_DEFAULT,
                             unroll=True, jit=True):
    """Orthogonalize kernel using spectral and Bjorck algorithm."""
    if u is None:
        u = jnp.mean(kernel, axis=0)  # use average of columns as initial vector.
    normalized_kernel, u, sigma = spectral_normalization(kernel, u, maxiter_spectral, unroll=unroll, jit=jit)
    orthogonal_kernel = bjorck_algorithm(normalized_kernel, maxiter_bjorck, unroll=unroll, jit=jit)
    return orthogonal_kernel, u, sigma


class BjorckDense(nn.Module):
    """Dense layer with orthogonality constraint enforced by Bjorck algorithm.
    
    If the kernel is square then it is orthogonal.
    If there is more rows than than columns then the columns are orthogonal.
    If there is more columns than rows then the rows are orthogonal.

    Orthogonalization is performed using Power Iteration and Bjorck algorithms.
    """
    features: int
    maxiter_spectral: Optional[int] = None
    maxiter_bjorck: Optional[int] = None
    train: Optional[bool] = None
    kernel_init: Callable = initializers.orthogonal()
    bias_init: Callable = initializers.zeros
    u_init: Callable = jax.random.normal

    @nn.compact
    def __call__(self, inputs,
                 train=None,
                 maxiter_spectral=None,
                 maxiter_bjorck=None):
        """Forward pass.
        
        Args:
            inputs: array of shape (B, f_1) with B the batch_size.
            train: whether to use perform orthogonalization.
            maxiter_spectral: maximum number of iterations for spectral normalization.
            maxiter_bjorck: maximum number of iterations for Bjorck's algorithm.

        Returns:
            outputs: array of shape (B, features)
        """

        # init params
        kernel_shape = (inputs.shape[-1], self.features)
        kernel = self.param('kernel', self.kernel_init, kernel_shape)
        bias = self.param('bias', self.bias_init, (1,self.features))

        # init mutable variables
        orthogonal_kernel = self.variable('lip', 'orthogonal_kernel', jnp.zeros, kernel_shape)
        u = self.variable('lip', 'u', (lambda shape: self.u_init(self.make_rng('lip'), shape)), (kernel_shape[-1],))
        sigma = self.variable('lip', 'sigma', lambda shape: jnp.zeros(shape), ())
        
        train = nn.merge_param('train', self.train, train)
        if train:
            maxiter_spectral = nn.merge_param('maxiter_spectral', self.maxiter_spectral, maxiter_spectral)
            maxiter_bjorck = nn.merge_param('maxiter_bjorck', self.maxiter_bjorck, maxiter_bjorck)
            ortho_ker, _u, _sigma = kernel_orthogonalization(kernel, u.value,
                                                             maxiter_spectral, maxiter_bjorck,
                                                             unroll=True, jit=True)
            # cache the value for future use.
            orthogonal_kernel.value = ortho_ker  
            u.value = _u
            sigma.value = _sigma
        else:
            ortho_ker = orthogonal_kernel.value

        y = jnp.matmul(inputs, ortho_ker)
        y = y + bias

        return y


def channelwise_groupsort2(x):
    """GroupSort2 activation function.

    Args:
        x: array of shape (C,). C must be an even number.

    Returns:
        array of shape (C,) with groupsort2 applied.
    """
    assert x.shape[0] % 2 == 0
    a, b = jnp.split(x, 2, axis=0)
    min_ab = jnp.minimum(a, b)
    max_ab = jnp.maximum(a, b)
    return jnp.concatenate([min_ab, max_ab], axis=0)


def groupsort2(x):
    """GroupSort2 activation function.

    Args:
        x: array of shape (B, C). C must be an even number.

    Returns:
        array of shape (B, C) with groupsort2 applied.
    """
    return jax.vmap(channelwise_groupsort2, in_axes=0)(x)
