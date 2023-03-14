# This file is part of CNQR which is released under the Apache License 2.0.
# See file LICENSE in the root directory or https://opensource.org/licenses/Apache-2.0 for full license details.

"""Implement of Lipschitz neural network layer in Jax.

Freely inspired by Deel.Lip package: https://github.com/deel-ai/deel-lip distributed under MIT licence.
"""

from functools import partial
from typing import Any, Callable, List
from typing import Optional, Sequence, Tuple, Union

import jax
from jax import lax
import jax.numpy as jnp
import jax.nn.initializers as initializers
from jax import eval_shape
from jax import lax
from jax import ShapedArray

import flax
import flax.linen as nn
from flax.linen.linear import canonicalize_padding
from flax.linen.linear import _conv_dimension_numbers
from flax.linen.pooling import pool

import numpy as onp

from cnqr._src.parametrizations import CachedParametrization
from cnqr.parametrizations import BjorckParametrization
from cnqr.parametrizations import Normalized2ToInftyParametrization, NormalizedInftyParametrization


# Macros from flax/linen/linear.py
PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str],
                      Tuple[lax.Precision, lax.Precision]]
DotGeneralT = Callable[..., Array]
ConvGeneralDilatedT = Callable[..., Array]


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
  def __call__(self, inputs: Array, train=None) -> Array:
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

    # orthogonalize
    stiefel_parametrization = self.stiefel_parametrization()
    ortho_ker = stiefel_parametrization(kernel, train=train)

    y = jnp.matmul(inputs, ortho_ker)
    if self.use_bias:
      bias = self.param('bias', self.bias_init, (self.features,))
      bias = jnp.reshape(bias, (1, self.features))
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
  def __call__(self, inputs: Array, train: bool = None) -> Array:
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

    normalization_parametrization = self.normalization_parametrization()
    normalized_kernel = normalization_parametrization(kernel, train=train)

    y = jnp.matmul(inputs, normalized_kernel)
    if self.use_bias:
      bias = self.param('bias', self.bias_init, (self.features,))
      bias = jnp.reshape(bias, (1, self.features))
      y = y + bias

    return y


Normalized2ToInftyDense = partial(NormalizedDense, normalization_parametrization=Normalized2ToInftyParametrization)
NormalizedInftyDense = partial(NormalizedDense, normalization_parametrization=NormalizedInftyParametrization)


##############################
##### Convolution layers #####
##############################


PaddingLike = Union[str, int, Sequence[Union[int, Tuple[int, int]]]]
LaxPadding = Union[str, Sequence[Tuple[int, int]]]


def RKO_initializer(init_2d: Callable) -> Callable:
  """Initializer for RKO kernel.
  
  Args:
    init_2d: initializer function for 2D kernel.
    
  Returns:
    initializer function for RKO kernel with the same signature as init_2d.
  """
  def init(key, shape, dtype=jnp.float_):
    shape_2d = shape[:-1] + (shape[-1],)  # flatten spatial dims.
    kernel_2d = init_2d(key, shape_2d, dtype)
    kernel = jnp.reshape(kernel_2d, shape)
    return kernel
  return init


class RKOConv(nn.Module):
  """Convolution layer with orthogonality constraint based on Reshaped Kernel Orthogonalization (RKO).

  The kernel K is orthogonal, but the associated convolution operator (*)_K is not necessarily orthogonal.

  The overlaping receptive fields of the convolution operator can cause loss of orthogonality, 
  and increase the Lipschitz constant of the operator. Hence we rescale the kernel to ensures a Lipschitz constant of 1.
  Unfortunately, some singular values of the convolution end-up being smaller than 1 in the process.
  Beware that this can cause vanishing gradients when too much convolutions are stacked.

  The reshaped kernel will be full rank only if features_in * spatial_dims = features_out. In this case implicit differentiation is possible.
    If features_in * spatial_dims < features_out, then the kernel may be rank-deficient in forward pass (collapse).
    If features_in * spatial_dims > features_out, then the kernel may be rank-deficient in backward pass (vanishing gradients).

  Warning: channels_last is the default, channels_first is not supported.
    Only Conv2D setting has been tested.
    Only a single batch dimension is supported.

  Attributes:
    features: number of convolution filters.
    kernel_size: shape of the convolutional kernel.
      For 1D convolution, the kernel size can be passed as an integer. For all other cases, it must be a sequence of integers.
    strides: an integer or a sequence of n integers, representing the inter-window strides (default: 1).
    padding: either the string 'SAME', the string 'VALID', the string 'CIRCULAR' (periodic boundary conditions),
      or a sequence of n (low, high) integer pairs that give the padding to apply before and after each spatial dimension.
      A single int is interpeted as applying the same padding in all dims and passign a single int in a sequence causes the same padding to be used on both sides.
      'CAUSAL' padding for a 1D convolution will left-pad the convolution axis, resulting in same-sized output.
    use_bias: whether to add a bias to the output (default: True).
    rko_method: method to orthogonalize the kernel. Default is BjorckParametrization.
    force_1lip: whether to rescale the kernel to ensure a Lipschitz constant of 1 (default: True).
    kernel_init: initializer for the convolutional kernel. Default is orthogonal initialization, using RKO wrapper.
    bias_init: initializer for the bias.
  """
  features: int
  kernel_size: Sequence[int]
  strides: Union[None, int, Sequence[int]] = 1
  padding: PaddingLike = 'SAME'
  kernel_dilation: Union[None, int, Sequence[int]] = 1
  use_bias: bool = True
  rko_method: CachedParametrization = BjorckParametrization
  force_1lip: bool = True
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = RKO_initializer(initializers.orthogonal())
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
  conv_general_dilated: ConvGeneralDilatedT = lax.conv_general_dilated

  def _get_conv_shapes(self):
    if isinstance(self.kernel_size, int):
      raise TypeError(f'Expected Conv kernel_size to be a tuple/list of integers (eg.: [3, 3]) but got {self.kernel_size}.')

    kernel_size = tuple(self.kernel_size)

    def maybe_broadcast(x: Optional[Union[int, Sequence[int]]]) -> (Tuple[int, ...]):
      if x is None:
        x = 1
      if isinstance(x, int):
        return (x,) * len(self.kernel_size)
      return tuple(x)

    # self.strides or (1,) * (inputs.ndim - 2)
    strides = maybe_broadcast(self.strides)
    kernel_dilation = maybe_broadcast(self.kernel_dilation)

    return kernel_size, strides, kernel_dilation

  def _lipschitz_factor(self, inputs_shape: Shape) -> float:
    """Compute the Lipschitz factor to apply on estimated Lipschitz constant in
    convolutional layer. This factor depends on the kernel size, the strides and the
    input shape.

    Warning: code has only been tested for images in input, i.e for Conv2D setting.
    No guarantees that it works in other settings.

    Copied from deel/lip/layers/convolutional.py version 1.4.0.
    See appendix of [1] for details.

    [1] Serrurier, Mathieu, Franck Mamalet, Alberto González-Sanz, Thibaut Boissin, Jean-Michel Loubes, et Eustasio del Barrio. 2021.
      « Achieving robustness in classification using optimal transport with hinge regularization ».
      arXiv. https://doi.org/10.48550/arXiv.2006.06520.
    """
    kernel_size, strides, _ = self._get_conv_shapes()

    strides = onp.array(strides, dtype=jnp.float32)
    prod_strides = onp.prod(strides)
    k_spatial = onp.array(kernel_size, dtype=jnp.float32)
    k_spatial_div2 = (k_spatial - 1.) / 2.
    input_spatial = onp.array(inputs_shape[-len(k_spatial)-1:-1], dtype=jnp.float32)  # remove batch and channel dims.

    # vectorized implementation of the formula given in deel/lip/layers/convolutional.py
    if prod_strides == 1:
      num = onp.prod(input_spatial)
      den = onp.prod(k_spatial * input_spatial - k_spatial_div2 * (k_spatial_div2 + 1))
    else:
      num = 1.
      den = onp.prod(onp.ceil(k_spatial / strides))

    factor = onp.sqrt(num / den)

    return factor

  def _reshaped_kernel_orthogonalization(self, kernel: Array, train: bool) -> Array:
    """Orthogonalize a convolution kernel by reshaping it into a 2D matrix and orthogonalizing it.

    The kernel is reshaped into a 2D matrix of shape (out_channels, in_channels * spatial_dims) and orthogonalized.
    The resulting matrix is reshaped back into the original kernel shape.
    The reshaping is done in a way that preserves the spatial locality of the kernel.

    Args:
      kernel: array of shape (..., in_channels, out_channels) where ... denotes the spatial dimensions.

    Returns:
      orthogonalized_kernel: array of shape (..., in_channels, out_channels).
    """
    kernel_2d = jnp.reshape(kernel, (-1, kernel.shape[-1]))
    rko_kernel_2d = self.rko_method()(kernel_2d, train=train)
    rko_kernel = jnp.reshape(rko_kernel_2d, kernel.shape)
    return rko_kernel

  @nn.compact
  def __call__(self, inputs: Array, train: bool = None) -> Array:
    """Applies a shared convolution to the inputs.

    Args:
      inputs: input data with dimensions (batch_dims, spatial_dims...,
        features). This is the channels-last convention, i.e. NHWC for a 2d
        convolution and NDHWC for a 3D convolution. Note: this is different from
        the input convention used by `lax.conv_general_dilated`, which puts the
        spatial dimensions last. Moreover a single (leading) batch dimension is supported (contrary to nn.Conv).

    Returns:
      The convolved data.
    """
    # developper note: the code is adapted from flax.nn.Conv, with the following changes:
    # - the kernel is reshaped into a 2D matrix and orthogonalized
    # - input is assumed to be channels-last and a single batch dimension is supported
    # - input_dilation is not supported
    # - arbitrary precision is not supported
    # - feature_group_count is not supported
    kernel_size, strides, kernel_dilation = self._get_conv_shapes()

    # pad input.
    padding_lax = canonicalize_padding(self.padding, len(kernel_size))
    if padding_lax == 'CIRCULAR':
      kernel_size_dilated = [(k - 1) * d + 1 for k, d in zip(kernel_size, kernel_dilation)]
      zero_pad: List[Tuple[int, int]] = [(0, 0)]
      pads = (zero_pad + [((k - 1) // 2, k // 2) for k in kernel_size_dilated] + [(0, 0)])
      inputs = jnp.pad(inputs, pads, mode='wrap')
      padding_lax = 'VALID'
    elif padding_lax == 'CAUSAL':
      if len(kernel_size) != 1:
        raise ValueError('Causal padding is only implemented for 1D convolutions.')
      left_pad = kernel_dilation[0] * (kernel_size[0] - 1)
      pads = [(0, 0), (left_pad, 0), (0, 0)]
      inputs = jnp.pad(inputs, pads)
      padding_lax = 'VALID'

    # convert channel_first to channel_last in order to use lax.conv_general_dilated.
    dimension_numbers = _conv_dimension_numbers(inputs.shape)
    in_features = jnp.shape(inputs)[-1]

    # orthogonalize the kernel and scale it.
    kernel_shape = kernel_size + (in_features, self.features)
    kernel = self.param('kernel', self.kernel_init, kernel_shape)
    rko_kernel = self._reshaped_kernel_orthogonalization(kernel, train=train)
    if self.force_1lip:
      rko_kernel = rko_kernel * self._lipschitz_factor(inputs.shape)

    y = self.conv_general_dilated(
        inputs,
        rko_kernel,
        strides,
        padding_lax,
        lhs_dilation=None,  # TODO: support input dilation (priority: low).
        rhs_dilation=kernel_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=1,  # TODO: support feature groups (priority: low). 
        precision=None,  # TODO: support arbitrary precision (priority: medium).
    )

    if self.use_bias:
      bias = self.param('bias', self.bias_init, (self.features,))
      bias = jnp.reshape(bias, (1,) * (y.ndim - 1) + (self.features,))
      y = y + bias

    return y


##########################################
############## Pooling ###################
##########################################


def l2norm_pool(inputs, window_shape, strides=None, padding="VALID"):
  """L2-norm pooling.

  Observe that this activation is Gradient Norm Preserving (GNP) 
  if strides == window_shape. This is the default behavior when 
  strides is None.

  Note: this function is not differentiable at 0. Beware of numerical errors.

  Args:
    inputs: input data with dimensions (batch, window dims..., features).
    window_shape: a shape tuple defining the window to reduce over.
    strides: a sequence of `n` integers, representing the inter-window
      strides (default: `window_shape`).
    padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
      of `n` `(low, high)` integer pairs that give the padding to apply before
      and after each spatial dimension (default: `'VALID'`).
    count_include_pad: a boolean whether to include padded tokens
      in the average calculation (default: `True`).

  Returns:
      array of shape (..., C) with l2norm pooling applied.
  """
  if strides is None:
    strides = window_shape
  epsilon = 1e-6  # like in deel-lip.
  y = jnp.square(inputs)
  y = pool(y, 0., lax.add, window_shape, strides, padding)
  # avoid issues in derivative of sqrt when y=0.
  y = jnp.sqrt(y + epsilon)  
  return y


def global_l2norm_pool(inputs):
  """Global L2-norm pooling.

  Observe that a single batch dimension is supported.

  Note: this function is not differentiable at 0. Beware of numerical errors.

  Args:
    inputs: input data with dimensions (batch, window dims..., features).

  Returns:
      array of shape (..., C) with global l2norm pooling applied.
  """
  y = l2norm_pool(inputs, inputs.shape[1:-1], strides=inputs.shape[1:-1], padding="VALID")
  y = jnp.reshape(y, (inputs.shape[0], -1))  # flatten spatial dimensions.
  return y


##############################################
############## Activations ###################
##############################################


def groupsort2(x):
  """GroupSort2 activation function along the last axis.

  Note: like in deel-lip library, the implementation uses a reshaping to ensure locality
  of the comparisons, which improves the memory access pattern and overall performance.

  Args:
      x: array of shape (..., C). C must be an even number. ... denotes the batch/data dimensions.

  Returns:
      array of shape (..., C) with groupsort2 applied.
  """
  n = x.shape[-1]
  if n%2 != 0:
    raise RuntimeError("The number of channels has to be an even number.")
  flat_x = jnp.reshape(x, (-1, n // 2, 2))
  a, b = jnp.split(flat_x, 2, axis=-1)
  min_ab = jnp.minimum(a, b)
  max_ab = jnp.maximum(a, b)
  flat_x = jnp.concatenate([min_ab, max_ab], axis=-1)
  return jnp.reshape(flat_x, x.shape)


def fullsort(x):
  """Full Sort activation function along the last axis.

  Args:
      x: array of shape (..., C).

  Returns:
      array of shape (..., C) with fullsort applied.
  """
  return jnp.sort(x, axis=-1)
