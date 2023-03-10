# This file is part of CNQR which is released under the Apache License 2.0.
# See file LICENSE in the root directory or https://opensource.org/licenses/Apache-2.0 for full license details.

"""Implement of Lipschitz neural network layer in Jax.

Freely inspired by Deel.Lip package: https://github.com/deel-ai/deel-lip distributed under MIT licence.
"""

from functools import partial
from re import T
from typing import Any, Callable, Iterable, List
from typing import Optional, Sequence, Tuple, Union

import jax
from jax import lax
import jax.numpy as jnp
import jax.nn.initializers as initializers
from jax.tree_util import tree_map

import jaxopt.projection
from jaxopt.tree_util import tree_zeros_like

import flax
import flax.linen as nn

from cnqr.loop import while_loop
from cnqr._src.parametrizations import CachedParametrization
from cnqr.parametrizations import BjorckParametrization
from cnqr.parametrizations import Normalized2ToInftyParametrization, NormalizedInftyParametrization


class StiefelDense(nn.Module):
  """Dense layer with orthogonality constraint enforced by projection onto Stiefel manifold.

  If the kernel is square then it is orthogonal.
  If there is more rows than than columns then the columns are orthogonal.
  If there is more columns than rows then the rows are orthogonal.

  Orthogonalization is performed byu default using Power Iteration and Bjorck algorithm.

  Attributes:
    features: number of output features.
    train: whether to train the layer (default: True).
    kernel_init: initializer for the weight matrix (default: orthogonal).
    bias_init: initializer for the bias (default: zero).
    stiefel_parametrization: parametrization of the Stiefel manifold (default: BjorckParametrization).
      note: changing default values of the parametrization is possible through the use of functools.partial.
  """
  features: int
  use_bias: bool = True
  kernel_init: Callable = initializers.orthogonal()
  bias_init: Callable = initializers.zeros
  stiefel_parametrization: CachedParametrization = BjorckParametrization

  @nn.compact
  def __call__(self, inputs, train=None):
    """Forward pass.

    Args:
      inputs: array of shape (B, f_1) with B the batch_size.
      train: whether to use perform orthogonalization of re-use the cached kernel.

    Returns:
      outputs: array of shape (B, features)
    """

    # init params
    kernel_shape = (inputs.shape[-1], self.features)
    kernel = self.param('kernel', self.kernel_init, kernel_shape)
    if self.use_bias:
      bias = self.param('bias', self.bias_init, (1, self.features))

    # orthogonalize
    stiefel_parametrization = self.stiefel_parametrization()
    ortho_ker = stiefel_parametrization(kernel, train=train)

    y = jnp.matmul(inputs, ortho_ker)
    if self.use_bias:
      y = y + bias

    return y


class NormalizedDense(nn.Module):
  """Dense layer with normalized matrix weights.
  
  Attributes:
    normalize_fun: function to normalize the matrix weights.
    features: number of output features.
    use_bias: whether to add a bias to the output (default: True).
    kernel_init: initializer for the kernel.
    bias_init: initializer for the bias.
  """
  features: int
  normalization_parametrization: CachedParametrization
  use_bias: bool = True
  kernel_init: Callable = initializers.lecun_normal()
  bias_init: Callable = initializers.zeros

  @nn.compact
  def __call__(self, inputs, train=None):
    """Forward pass.

    Args:
      inputs: array of shape (B, f_1) with B the batch_size.
      train: whether to use perform orthogonalization of re-use the cached kernel.

    Returns:
      outputs: array of shape (B, features)
    """

    # init params
    kernel_shape = (inputs.shape[-1], self.features)
    kernel = self.param('kernel', self.kernel_init, kernel_shape)

    if self.use_bias:
      bias = self.param('bias', self.bias_init, (1, self.features))

    normalization_parametrization = self.normalization_parametrization()
    normalized_kernel = normalization_parametrization(kernel, train=train)

    y = jnp.matmul(inputs, normalized_kernel)
    if self.use_bias:
      y = y + bias

    return y


Normalized2ToInftyDense = partial(NormalizedDense, normalization_parametrization=Normalized2ToInftyParametrization)
NormalizedInftyDense = partial(NormalizedDense, normalization_parametrization=NormalizedInftyParametrization)

##############################################
############## Activations ###################
##############################################


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


def fullsort(x):
  """FullSort activation function.

  Args:
      x: array of shape (B, C).

  Returns:
      array of shape (B, C) with full sort applied: dimension C is sorted.
  """
  return jnp.sort(x, axis=-1)
