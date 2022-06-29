"""Implement of Lipschitz neural network layer."""


from typing import Any, Callable, Iterable, List
from typing import Optional, Sequence, Tuple, Union

import jax
from jax import lax
import jax.numpy as jnp
import jax.nn.initializers as initializers
import flax
import flax.linen as nn

from loop import while_loop


def l2_normalize(vec, epsilon=1e-5):
    norm = jnp.linalg.norm(vec) + epsilon
    return vec / norm


def power_iteration(kernel, u_init, maxiter_spectral,
                    unroll=True, jit=True):
    """Power iteration for finding the largest eigenvalue and eigenvector."""
    
    def body_fun(t):
        u, v = t
        v = l2_normalize(jnp.matmul(kernel, u))
        u = l2_normalize(jnp.matmul(v, kernel))
        return u, v
    
    def cond_fun(t):
        del t
        return True

    v_init = l2_normalize(jnp.matmul(kernel, u_init))
    u, v = while_loop(body_fun,
                      lambda _: True, (u_init, v_init),
                      maxiter=maxiter_spectral,
                      unroll=unroll, jit=jit)

    return u, v


def spectral_normalization(kernel, u, maxiter_spectral):
    """Spectral normalization of kernel."""
    u, v = power_iteration(kernel, u, maxiter_spectral)
    sigma = jnp.dot(v, jnp.dot(kernel, u))
    unit_spectral_norm_kernel = kernel / sigma
    return unit_spectral_norm_kernel, u


def bjorck_algorithm(kernel, maxiter_bjorck, beta=0.5,
                     unroll=True, jit=True):
    """Bjorck algorithm for finding the orthogonal kernel.

    kernel must be normalized.
    
    Args:
        kernel: array of shape (A, B)
        maxiter_bjorck: maximum number of iterations
        beta: beta parameter for the algorithm (default: 0.5)
        unroll: whether to unroll the loop (default: False)
        jit: whether to use jit (default: True)
        
    Returns:
        orthogonal kernel
    """
    
    def body_fun(kernel):
        _32_W = (1 + beta) * kernel
        _12_WWW = beta * kernel @ kernel.T @ kernel
        return _32_W - _12_WWW

    orthogonal_kernel = while_loop(body_fun,
                                   lambda _: True, kernel,
                                   maxiter=maxiter_bjorck,
                                   unroll=unroll, jit=jit)
    
    return orthogonal_kernel


def kernel_orthogonalization(kernel, u,
                             maxiter_spectral, maxiter_bjorck,
                             unroll, jit):
    """Orthogonalize kernel using spectral and Bjorck algorithm."""
    normalized_kernel, u = spectral_normalization(kernel, u, maxiter_spectral, unroll=unroll, jit=jit)
    orthogonal_kernel = bjorck_algorithm(normalized_kernel, u, maxiter_bjorck, unroll=unroll, jit=jit)
    return orthogonal_kernel, u


class SpectralDense(nn.Module):
    """Spectral Dense layer."""
    features: int
    kernel_init: Callable = initializers.lecun_normal()
    bias_init: Callable = initializers.zeros
    u_init: Callable = initializers.lecun_normal()

    @nn.compact
    def __call__(self, inputs, training):
        """Forward pass.
        
        Args:
            inputs: array of shape (B, f) with B the batch_size.
            training: whether to use training mode.
        """

        # init params
        kernel_shape = (inputs.shape[-1], self.features)
        kernel = self.param('kernel', self.kernel_init, kernel_shape)
        bias = self.param('bias', self.bias_init, (self.features,))

        # init mutable variables
        orthogonal_kernel = self.variable('lip', 'orthogonal_kernel', initializers.zeros, kernel_shape)
        u = self.variable('lip', 'u', self.u_init, (kernel_shape[-1],))
        sigma = self.variable('lip', 'sigma', initializers.zeros, (1,))
        
        if training:
            ortho_ker, _u = kernel_orthogonalization(kernel, u.value, unroll=True, jit=True)
            orthogonal_kernel.value = ortho_ker  # assigns the value
            u.value = _u
        else:
            ortho_ker = orthogonal_kernel.value

        y = jnp.matmul(inputs, ortho_ker)
        y = y + bias

        return y
