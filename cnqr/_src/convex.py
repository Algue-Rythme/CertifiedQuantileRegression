# This file is part of CNQR which is released under the Apache License 2.0.
# See file LICENSE in the root directory or https://opensource.org/licenses/Apache-2.0 for full license details.

"""Implementation of Amos+(2017) partially input convex neural networks (PICNN)
with (slightly modified) initialization schemes proposed in Bunne+(2022).

Largely inspired by ott-jax package: https://github.com/ott-jax/ott distributed under Apache license.
"""
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn import initializers
from jax import random

from ott.solvers.nn.layers import PosDefPotentials

from cnqr._src.parametrizations import CachedParametrization
from cnqr.parametrizations import PositiveOrthant 


Shape = Tuple[int]
Dtype = Any
Array = Any
KeyArray = random.KeyArray
DTypeLikeInexact = Any  # DTypeLikeFloat | DTypeLikeComplex


def inv_act_fun_initializer(input_dims, inv_act_fun):
  """Initializer for the inverse softplus function.

  Return a function with signature `init(key, shape, dtype=jnp.float_) -> Array`

  Args:
    input_dims: (integer) size of input dimension.
    inv_act_fun: inverse of the activation function.
  """
  constant = inv_act_fun(1.0 / input_dims)
  constant_init = initializers.constant(constant)
  return constant_init


@dataclass
class PositiveDense(nn.Module):
  """Dense layer with normalized matrix weights.
  
  Attributes:
    normalize_fun: function to normalize the matrix weights.
    features: number of output features.
    use_bias: whether to add a bias to the output (default: True).
    kernel_init: initializer for the kernel.
    bias_init: initializer for the bias.
  """
  features: int
  use_bias: bool = True
  kernel_init: Union[Callable, str] = initializers.lecun_normal()
  bias_init: Callable = initializers.zeros
  positive_parametrization: CachedParametrization = PositiveOrthant

  @nn.compact
  def __call__(self, inputs: Array, train: bool = None) -> Array:
    """Forward pass.

    Args:
      inputs: array of shape (B, f_1) with B the batch_size.
      train: whether to use perform orthogonalization of re-use the cached kernel.

    Returns:
      outputs: array of shape (B, features)
    """
    positive_param = self.positive_parametrization()

    input_dims = inputs.shape[-1]
    kernel_init = self.kernel_init
    if isinstance(kernel_init, str) and kernel_init == 'inv_act_fun':
      kernel_init = inv_act_fun_initializer(input_dims, positive_param.inv_act_fun)

    # init params
    kernel_shape = (input_dims, self.features)
    kernel = self.param('kernel', kernel_init, kernel_shape)

    positive_kernel = positive_param(kernel, train=train)

    y = jnp.matmul(inputs, positive_kernel)
    if self.use_bias:
      bias = self.param('bias', self.bias_init, (self.features,))
      bias = jnp.reshape(bias, (1, self.features))
      y = y + bias

    return y


class DenseWrapper(nn.Module):
  features: int
  name: str = 'dense'
  use_bias: bool = True
  bias_init_value: float = 0.0
  kernel_init_type: float = 'lecun_normal'  # alternatives: ['normal', 'identity', 'constant']
  kernel_epsilon_init: float = 1e-2
  kernel_constant_init: float = 1.0
  layer_class: nn.Module = nn.Dense  # alternative: PositiveDense

  @staticmethod
  def _identity_initializer(key: KeyArray,
                            shape: Shape,
                            dtype: DTypeLikeInexact = jnp.float_) -> Array:
    return jnp.eye(N=shape[-2], M=shape[-1], dtype=dtype).reshape(shape)

  @nn.compact
  def __call__(self, x, **kwargs):
    # definine initialization of weights and biases
    kernel_init_dict = {
      'lecun_normal': nn.initializers.lecun_normal(),
      'normal': nn.initializers.normal(self.kernel_epsilon_init),
      'identity': self._identity_initializer,
      'constant': initializers.constant(self.kernel_constant_init)
    }
    kernel_init = kernel_init_dict[self.kernel_init_type]

    f = self.layer_class(
      self.features,
      name=self.name,
      use_bias=self.use_bias,
      kernel_init=kernel_init,
      bias_init=nn.initializers.constant(self.bias_init_value),
      **kwargs
    )
    return f(x)


class PICNN(nn.Module):
  """Partially input convex neural network (PICNN) architecture with initialization.
  Implementation of partially input convex neural networks as introduced in
  Amos+(2017) with initialization schemes proposed by Bunne+(2022).

  Args:
    * dim_hidden: sequence specifying size of hidden dimensions. The
    output dimension of the last layer is 1 by default.
    * dim_y: data dimensionality (default: 2).
    * init_fn: choice of initialization method for weight matrices (default:
    `jax.nn.initializers.normal`).
    * epsilon_init: value of standard deviation of weight initialization method.
    * tau_act_fn: choice of activation function used in context network architecture
    * sigma_act_fn: choice of activation function used in convex network architecture
    * pos_act_fn: choice of positive activation function (default: `nn.relu`)
    * context_dense_layer: choice of layer type for the context modules
    (default: nn.Dense, later BjorckDense might be used)
    * gaussian_map: data inputs of source and target measures for
    initialization scheme based on Gaussian approximation of input and
    target measure (if None, identity initialization is used).
  """

  dim_hidden: Sequence[int]
  dim_y: int = 2
  init_fn: Callable = jax.nn.initializers.normal
  epsilon_init: float = 1e-2
  tau_act_fn: Callable = nn.softplus  # alternative -> GroupSort
  sigma_act_fn: Callable = nn.softplus  # alternative -> ReLU, ELU
  pos_act_fn: Callable = nn.relu
  context_dense_layer: nn.Module = nn.Dense  # TODO: try with BjorckDense
  gaussian_map: Tuple[jnp.ndarray, jnp.ndarray] = None

  @nn.compact
  def __call__(self, x, y):
    """Forward pass.

    Args:
      x: context variable.
        array of shape (B, n_features_x) with B the batch_size.
      y: observation from source distribution.
        array of shape (B, n_features_y) with B the batch_size.

    Returns:
      outputs: array of shape (B,)
    """
    num_hidden = len(self.dim_hidden)

    # ===== Input encoding =====
    # encoding the input of context network (upscale to dim_hidden[0])
    phi = DenseWrapper(
      features=self.dim_hidden[0],
      name='phi_dense',
      kernel_init_type='lecun_normal',
      use_bias=True,
      bias_init_value=0.0
    )

    # encoding the input of convex network (quadratic potential of y)
    q = PosDefPotentials(
      self.dim_y,
      name='q_potential',
      num_potentials=1,
      kernel_init=DenseWrapper._identity_initializer,
      bias_init=nn.initializers.zeros,
      use_bias=True,
    )

    # ===== Context network =====
    # (now simple dense, then will be replaced with lipschitz)
    # dense layers (with bias), init: weights -> identity, bias -> 0
    f_x = partial(
      DenseWrapper,
      kernel_init_type='identity',
      use_bias=True,
      bias_init_value=0
    )
    output_dims_x = self.dim_hidden

    # ===== Context modulators =====
    # f_u: weights init -> approx(0), bias init -> 1.0
    f_u = partial(
      DenseWrapper,
      kernel_init_type='normal',
      kernel_epsilon_init=self.epsilon_init,
      use_bias=True,
      bias_init_value=1.0
    )
    output_dims_u = [1] + list(self.dim_hidden)[1:] + [1]

    # f_v: weights init -> approx(0), bias init -> 0.0
    f_v = partial(
      DenseWrapper,
      kernel_init_type='normal',
      kernel_epsilon_init=self.epsilon_init,
      use_bias=True,
      bias_init_value=0.0
    )
    output_dims_v = [self.dim_y] * num_hidden + [1]

    # f_w: weights init -> approx(0), bias init -> 0.0
    f_w = partial(
      DenseWrapper,
      kernel_init_type='normal',
      kernel_epsilon_init=self.epsilon_init,
      use_bias=True,
      bias_init_value=0.0
    )
    output_dims_w = list(self.dim_hidden) + [1]

    # ===== Convex network =====
    # w_U:  positive dense layers (no bias), weights init -> ones / d1
    w_U = partial(
      DenseWrapper,
      layer_class=PositiveDense,
      kernel_init_type='constant',
      use_bias=False,
    )
    rescale = PositiveDense.inv_rectifier_fn
    kernel_constant_inits_U = [rescale(1.0 / d1) for d1 in [1] + list(self.dim_hidden)]
    output_dims_U = list(self.dim_hidden) + [1]

    # w_V: dense layers (no bias), weights init -> approx(0)
    w_V = partial(
      DenseWrapper,
      kernel_init_type='normal',
      kernel_epsilon_init=self.epsilon_init,
      use_bias=False,
    )
    output_dims_V = list(self.dim_hidden) + [1]

    # ===== Inference =====
    x = phi(x)
    z = q(y).reshape(x.shape[0], -1)
    num_steps = len(self.dim_hidden) + 1
    for i in range(num_steps):
      # context modulators
      u = f_u(features=output_dims_u[i], name=f'f_u_dense_{i}')(x)
      v = f_v(features=output_dims_v[i], name=f'f_v_dense_{i}')(x)
      w = f_w(features=output_dims_w[i], name=f'f_w_dense_{i}')(x)
      # convex network
      w_Uk = w_U(features=output_dims_U[i], name=f'w_U_pdense_{i}',
                 kernel_constant_init=kernel_constant_inits_U[i])
      w_Vk = w_V(features=output_dims_V[i], name=f'w_V_dense_{i}')
      h = w_Uk(z * self.pos_act_fn(u)) + w_Vk(y * v) + w
      z = self.sigma_act_fn(h)
      # context network
      if i < num_steps - 1:
        x = f_x(features=output_dims_x[i], name=f'f_x_dense_{i}')(x)
        x = self.tau_act_fn(x)
    return z.reshape(-1,)
