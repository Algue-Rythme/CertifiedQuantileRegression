# This file is part of CNQR which is released under the Apache License 2.0.
# See file LICENSE in the root directory or https://opensource.org/licenses/Apache-2.0 for full license details.

"""Test Lipschitz neural network layers.

Tests are borrowed from Deel-Lip library.
"""
from functools import partial
from typing import Sequence

from absl.testing import parameterized
from absl.testing import absltest

import jax
import jax.numpy as jnp
import jax._src.test_util as jtu
import flax.linen as nn
import numpy as onp

from scipy.linalg import norm, eig

import cnqr.layers as layers
import cnqr.tree_util as tree_util
from cnqr.parametrizations import BjorckParametrization
from cnqr.layers import StiefelDense, RKOConv, NormalizedInftyDense, Normalized2ToInftyDense
from cnqr.layers import groupsort2, full_sort


"""
About these tests:

==================
What is tested:
---------------
- layer instantiation
- prediction
- k lip_constraint is respected (at +-0.001 or more)

What is not tested:
-------------------
- training
- storing on disk and reloading
- layer performance (time / accuracy)
- layer structure (doesn't check that RKOConv is actually a convolution with the right padding)
"""


class StiefelDenseTest(jtu.JaxTestCase):

  def test_stiefel_dense_glorot_normal(self):
    """Test Stiefel projection for dense matrices with glorot normal initializer."""
    key = jax.random.PRNGKey(9874)
    key_params, key_lip, batch_1_key, batch_2_key = jax.random.split(key, 4)

    batch_size = 16
    input_size = 5
    batch_1 = jax.random.normal(key=batch_1_key, shape=(batch_size, input_size))
    batch_2 = jax.random.normal(key=batch_2_key, shape=(batch_size, input_size))

    features = 15
    model = StiefelDense(features=features,
                         kernel_init=jax.nn.initializers.glorot_normal())
    params = model.init(rngs={'params': key_params, 'lip': key_lip}, inputs=batch_1, train=True)

    y_1 = model.apply(params, batch_1, train=False)
    y_2 = model.apply(params, batch_2, train=False)

    y1_y2 = y_1[:, jnp.newaxis, :] - y_2[jnp.newaxis, :, :]
    y1_y2_norm = jnp.linalg.norm(y1_y2, axis=-1)

    x1_x2 = batch_1[:, jnp.newaxis, :] - batch_2[jnp.newaxis, :, :]
    x1_x2_norm = jnp.linalg.norm(x1_x2, axis=-1)

    atol = 1e-4
    rtol = 1e-4

    is_1_lip = (y1_y2_norm < (x1_x2_norm + atol)).flatten().tolist()
    self.assertTrue(all(is_1_lip))

    # compare distances between all pairs of points in the batch.
    # since features >> input_size, the distance between points in the batch
    # should be close to the distance between the points in the input space.
    self.assertAllClose(y1_y2_norm, x1_x2_norm, atol=atol, rtol=rtol)

  def test_stiefel_dense_mutable(self):
    """Test Stiefel projection for dense matrices with mutable variables."""
    key = jax.random.PRNGKey(444)
    key_params, key_lip, batch_1_key, batch_2_key = jax.random.split(key, 4)

    batch_size = 16
    input_size = 5
    batch_1 = jax.random.normal(key=batch_1_key, shape=(batch_size, input_size))
    batch_2 = jax.random.normal(key=batch_2_key, shape=(batch_size, input_size))

    features = 15
    model = StiefelDense(features=features)
    params = model.init(rngs={'params': key_params, 'lip': key_lip}, inputs=batch_1, train=True)

    y_1, mutated_1 = model.apply(params, batch_1, train=True, mutable=['lip'])
    y_2, mutated_2 = model.apply(params, batch_2, train=True, mutable=['lip'])

    y1_y2 = y_1[:, jnp.newaxis, :] - y_2[jnp.newaxis, :, :]
    y1_y2_norm = jnp.linalg.norm(y1_y2, axis=-1)

    x1_x2 = batch_1[:, jnp.newaxis, :] - batch_2[jnp.newaxis, :, :]
    x1_x2_norm = jnp.linalg.norm(x1_x2, axis=-1)

    atol = 1e-4
    rtol = 1e-4

    is_1_lip = (y1_y2_norm < (x1_x2_norm + atol)).flatten().tolist()
    self.assertTrue(all(is_1_lip))

    # compare distances between all pairs of points in the batch.
    # since features >> input_size, the distance between points in the batch
    # should be close to the distance between the points in the input space.
    self.assertAllClose(y1_y2_norm, x1_x2_norm, atol=atol, rtol=rtol)

    parametrization_name = list(mutated_1['lip'].keys())[0]
    lip_vars_1 = mutated_1['lip'][parametrization_name]
    lip_vars_2 = mutated_2['lip'][parametrization_name]
    self.assertAllClose(lip_vars_1['u'], lip_vars_2['u'], atol=atol, rtol=rtol)

  def test_stiefel_dense_unrolled(self):
    key = jax.random.PRNGKey(444)
    key_params, key_lip, batch_key = jax.random.split(key, 3)

    batch_size = 16
    input_size = 5
    batch = jax.random.normal(key=batch_key, shape=(batch_size, input_size))

    features = 15
    implicit_bjorck = partial(BjorckParametrization, auto_diff='unroll')
    model = StiefelDense(features=features, stiefel_parametrization=implicit_bjorck)
    all_params = model.init(rngs={'params': key_params, 'lip': key_lip}, inputs=batch, train=True)

    params = all_params['params']
    params_lip = all_params['lip']

    def forward_model(params, inputs):
      preds, _ = model.apply({'params': params, 'lip': params_lip}, inputs, train=True, mutable=['lip'])
      loss = jnp.mean(preds)
      return loss

    eps = 1e-3
    rtol = 1e-4
    jtu.check_grads(forward_model, (params, batch), modes=['rev'], order=1, eps=eps, rtol=rtol)

  def test_stiefel_dense_implicit(self):
    key = jax.random.PRNGKey(444)
    key_params, key_lip, batch_key = jax.random.split(key, 3)

    batch_size = 16
    input_size = 50
    batch = jax.random.normal(key=batch_key, shape=(batch_size, input_size))

    features = input_size
    implicit_bjorck = partial(BjorckParametrization, auto_diff='implicit')
    model = StiefelDense(features=features, stiefel_parametrization=implicit_bjorck)
    all_params = model.init(rngs={'params': key_params, 'lip': key_lip}, inputs=batch, train=True)

    params = all_params['params']
    params_lip = all_params['lip']

    def forward_model(params, inputs):
      preds, _ = model.apply({'params': params, 'lip': params_lip}, inputs, train=True, mutable=['lip'])
      loss = jnp.mean(preds)
      return loss

    eps = 1e-3
    rtol = 1e-4
    jtu.check_grads(forward_model, (params, batch), modes=['rev'], order=1, eps=eps, rtol=rtol)

  def test_dense_architecture(self):
    """Test Sequential Dense Neural Network end-to-end."""
    key = jax.random.PRNGKey(822)
    key_params, key_lip, batch_1_key, batch_2_key = jax.random.split(key, 4)

    batch_size = 1  # mandatory for this test.
    input_size = 10
    batch_1 = jax.random.normal(key=batch_1_key, shape=(batch_size, input_size))
    batch_2 = jax.random.normal(key=batch_2_key, shape=(batch_size, input_size))

    class Sequential(nn.Module):  # nn.Sequential is not compatible with train=True.
      features : Sequence[int]
      @nn.compact
      def __call__(self, inputs, train=None):
        x = inputs
        for num_features in self.features:
          x = StiefelDense(num_features)(x, train=train)
        return x
    
    model = Sequential([32, 48, 4, 1])
    params = model.init(rngs={'params': key_params, 'lip': key_lip}, inputs=batch_1, train=True)

    y_1, mutated_1 = model.apply(params, batch_1, train=True, mutable=['lip'])
    y_2, mutated_2 = model.apply(params, batch_2, train=True, mutable=['lip'])

    L = jnp.abs(y_1 - y_2) / jnp.linalg.norm(batch_1 - batch_2)

    # Test that the Lipschitz constant is less than 1 (up to numerical error).
    atol = 1e-4
    rtol = 1e-4
    self.assertLessEqual(L, 1. + atol)

    tree_util.tree_map(lambda a, b: self.assertAllClose(a, b, atol=atol, rtol=rtol), mutated_1['lip'], mutated_2['lip'])


class RKOConvTest(jtu.JaxTestCase):

  @parameterized.parameters(((1, 1), (1, 1), 'SAME'),
                            ((3, 3), (1, 1), 'VALID'),
                            ((3, 3), (2, 2), 'CIRCULAR'))
  def test_conv2d(self, kernel_size, strides, padding):
    """Test RKOConvolution with 2D inputs."""
    key = jax.random.PRNGKey(231)
    key_params, key_lip, batch_key = jax.random.split(key, 3)

    batch_size, input_height, input_width, num_channels = 6, 28, 36, 3
    batch = jax.random.uniform(key=batch_key, shape=(batch_size, input_height, input_width, num_channels),
                               minval=0., maxval=1.)

    features = 12
    model = RKOConv(features=features, kernel_size=kernel_size, strides=strides, padding=padding)
    all_params = model.init(rngs={'params': key_params, 'lip': key_lip}, inputs=batch, train=True)

    params = all_params['params']
    params_lip = all_params['lip']

    epsilon = 1e-7
    def forward_model(params, inputs):
      preds, _ = model.apply({'params': params, 'lip': params_lip}, inputs, train=True, mutable=['lip'])
      loss = jnp.sqrt(jnp.sum(preds ** 2) + epsilon)  # Norm Preserving loss => Gradient Norm Preserving operation.
      return loss

    # batch computation of gradients norms wrt to each input.
    Jxf = jax.vmap(jax.grad(forward_model, argnums=1), in_axes=(None, 0))(params, batch[:, None, ...])
    Jxf = jnp.reshape(Jxf, (batch_size, -1))
    Jxf_norm = jnp.linalg.norm(Jxf, axis=1)

    # check that the Lipschitz constant is less than 1 (up to numerical error).
    atol = 1e-4
    self.assertLessEqual(jnp.max(Jxf_norm), 1. + atol)

    # Check that the 1x1 convolution is gradient norm preserving.
    if kernel_size == (1, 1) and strides == (1, 1):
      self.assertLessEqual(1. - atol, jnp.min(Jxf_norm))

    # check that the gradient is computed correctly.
    eps = 1e-3
    rtol = 1e-3
    jtu.check_grads(forward_model, (params, batch), modes=['rev'], order=1, eps=eps, rtol=rtol)


class ActivationsTest(jtu.JaxTestCase):

  def test_groupsort2(self):
    """Test GroupSort2."""
    vec = jnp.array([[4, 1, 5, 8, 3, 2, 6, 7]])
    vec_group = groupsort2(vec)
    answer = jnp.array([[1, 4, 5, 8, 2, 3, 6, 7]])
    onp.testing.assert_array_equal(vec_group, answer)

  def test_groupsort2_batch(self):
    """Test GroupSort2."""
    vec = jnp.array([[[4, 1, 5, 8, 3, 2, 6, 7], [10, 20, 30, 50, 40, 60, 70, 80]],
                     [[8, 7, 6, 5, 4, 3, 2, 1], [ 1,  1,  0,  1,  0,  0,  1,  0]]])
    vec_group = groupsort2(vec)
    answer = jnp.array([[[1, 4, 5, 8, 2, 3, 6, 7], [10, 20, 30, 50, 40, 60, 70, 80]],
                        [[7, 8, 5, 6, 3, 4, 1, 2], [ 1,  1,  0,  1,  0,  0,  0,  1]]])
    onp.testing.assert_array_equal(vec_group, answer)

  def test_fullsort_batch(self):
    """Test FullSort."""
    vec = jnp.array([[[4, 1, 5, 8, 3, 2, 6, 7], [10, 20, 30, 50, 40, 60, 70, 80]],
                     [[8, 7, 6, 5, 4, 3, 2, 1], [ 1,  1,  0,  1,  0,  0,  1,  0]]])
    vec_group = full_sort(vec)
    answer = jnp.array([[[1, 2, 3, 4, 5, 6, 7, 8], [10, 20, 30, 40, 50, 60, 70, 80]],
                        [[1, 2, 3, 4, 5, 6, 7, 8], [ 0,  0,  0,  0,  1,  1,  1,  1]]])
    onp.testing.assert_array_equal(vec_group, answer)


if __name__ == '__main__':
  jax.config.update("jax_debug_nans", True)
  jax.config.update("jax_enable_x64", False)
  absltest.main()
