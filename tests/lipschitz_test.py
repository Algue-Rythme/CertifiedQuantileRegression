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
from cnqr.layers import StiefelDense, groupsort2


class BjorckDenseTest(jtu.JaxTestCase):

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

  def test_lipschitz_architecture(self):
    """Lipschitz Neural Network."""
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


class ActivationsTest(jtu.JaxTestCase):

  def test_groupsort2(self):
    """Test GroupSort2."""
    vec = jnp.array([[1, 4, 5, 8] + [2, 3, 6, 7]])
    vec_group = groupsort2(vec)
    answer = jnp.array([[1, 3, 5, 7] + [2, 4, 6, 8]])
    onp.testing.assert_array_equal(vec_group, answer)


if __name__ == '__main__':
  jax.config.update("jax_debug_nans", True)
  jax.config.update("jax_enable_x64", False)
  absltest.main()
