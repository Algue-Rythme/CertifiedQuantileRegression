# This file is part of CNQR which is released under the Apache License 2.0.
# See file LICENSE in the root directory or https://opensource.org/licenses/Apache-2.0 for full license details.

"""Test Lipschitz neural network layers.

Tests are borrowed from Deel-Lip library.
"""
from functools import partial

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
from cnqr._src.lipschitz import BjorckDense, power_iteration, spectral_normalization
from cnqr._src.lipschitz import l2_normalize, bjorck_algorithm, kernel_orthogonalization
from cnqr._src.lipschitz import groupsort2


class BjorckDenseTest(jtu.JaxTestCase):

  @parameterized.parameters(((20, 10),), ((30, 30),), ((10, 60),))
  def test_power_iteration(self, kernel_shape):
    """Test power iteration forward."""
    onp.random.seed(6813)

    maxiter_spectral = 100

    kernel = onp.random.normal(size=kernel_shape)
    u_init = onp.random.normal(size=(kernel_shape[-1],))
    u, v = power_iteration(kernel, u_init, maxiter_spectral=maxiter_spectral, unroll=True, jit=True)
    sigma = jnp.dot(v, jnp.dot(kernel, u))

    sigma_scipy = norm(kernel, ord=2)

    atol = 1e-4
    rtol = 1e-4
    self.assertAllClose(norm(u, ord=2), 1., atol=atol, rtol=rtol)
    self.assertAllClose(norm(v, ord=2), 1., atol=atol, rtol=rtol)
    self.assertAllClose(sigma, sigma_scipy, atol=atol, rtol=rtol)

  @parameterized.parameters(((20, 10),), ((30, 30),), ((10, 60),))
  def test_spectral_normalization(self, kernel_shape):
    """Test spectral normalization forward."""
    onp.random.seed(967)

    maxiter_spectral = 100

    kernel = onp.random.normal(size=kernel_shape)
    u_init = onp.random.normal(size=(kernel_shape[-1],))
    ortho, u, sigma = spectral_normalization(kernel, u_init, maxiter_spectral=maxiter_spectral, unroll=True, jit=True)

    K_hat_u = jnp.matmul(ortho, u)
    K_u = jnp.matmul(kernel, u)

    atol = 1e-4
    rtol = 1e-4
    self.assertAllClose(K_hat_u * sigma, K_u, atol=atol, rtol=rtol)

  @parameterized.parameters(((20, 10),), ((30, 30),), ((10, 60),))
  def test_kernel_orthogonalization(self, kernel_shape):
    """Test kernel orthogonalization of arbitrary matrices."""
    onp.random.seed(9874)

    kernel = onp.random.normal(size=kernel_shape)
    u_init = onp.random.normal(size=(kernel_shape[-1],))

    maxiter_spectral = 100
    maxiter_bjorck = 50

    K, u, sigma = kernel_orthogonalization(kernel, u_init,
                                           maxiter_spectral=maxiter_spectral,
                                           maxiter_bjorck=maxiter_bjorck,
                                           unroll=False, jit=True)
    if kernel_shape[0] > kernel_shape[1]:
      gram_K = jnp.matmul(K.T, K)
    else:
      gram_K = jnp.matmul(K, K.T)

    atol = 1e-4
    rtol = 1e-4
    self.assertAllClose(gram_K, jnp.eye(*gram_K.shape), atol=atol, rtol=rtol)

  def test_bjorck_dense_glorot_normal(self):
    """Test Bjorck algorithm for dense matrices with glorot normal initializer."""
    key = jax.random.PRNGKey(9874)
    key_params, key_lip, batch_1_key, batch_2_key = jax.random.split(key, 4)

    batch_size = 16
    input_size = 5
    batch_1 = jax.random.normal(key=batch_1_key, shape=(batch_size, input_size))
    batch_2 = jax.random.normal(key=batch_2_key, shape=(batch_size, input_size))

    maxiter_spectral = 100
    maxiter_bjorck = 50

    features = 15
    model = BjorckDense(features=features,
                        kernel_init=jax.nn.initializers.glorot_normal(),
                        maxiter_spectral=maxiter_spectral,
                        maxiter_bjorck=maxiter_bjorck)
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
    assert all(is_1_lip)
    self.assertAllClose(y1_y2_norm, x1_x2_norm, atol=atol, rtol=rtol)

  def test_bjorck_dense_mutable(self):
    """Test Bjorck algorithm for dense matrices with mutable variables."""
    key = jax.random.PRNGKey(444)
    key_params, key_lip, batch_1_key, batch_2_key = jax.random.split(key, 4)

    batch_size = 16
    input_size = 5
    batch_1 = jax.random.normal(key=batch_1_key, shape=(batch_size, input_size))
    batch_2 = jax.random.normal(key=batch_2_key, shape=(batch_size, input_size))

    maxiter_spectral = 100
    maxiter_bjorck = 50

    features = 15
    model = BjorckDense(features=features,
                        maxiter_spectral=maxiter_spectral,
                        maxiter_bjorck=maxiter_bjorck)
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
    assert all(is_1_lip)
    self.assertAllClose(y1_y2_norm, x1_x2_norm, atol=atol, rtol=rtol)

    self.assertAllClose(mutated_1['lip']['u'], mutated_2['lip']['u'], atol=atol, rtol=rtol)

  def test_lipschitz_architecture(self):
    """Lipschitz Neural Network."""
    key = jax.random.PRNGKey(822)
    key_params, key_lip, batch_1_key, batch_2_key = jax.random.split(key, 4)

    batch_size = 1
    input_size = 10
    batch_1 = jax.random.normal(key=batch_1_key, shape=(batch_size, input_size))
    batch_2 = jax.random.normal(key=batch_2_key, shape=(batch_size, input_size))

    maxiter_spectral = 100
    maxiter_bjorck = 50

    features = [32, 48, 4, 1]
    bjorck_dense = partial(BjorckDense, maxiter_spectral=maxiter_spectral, maxiter_bjorck=maxiter_bjorck, train=True)
    model = nn.Sequential([bjorck_dense(num_features) for num_features in features])
    params = model.init(rngs={'params': key_params, 'lip': key_lip}, inputs=batch_1)

    y_1, mutated_1 = model.apply(params, batch_1, mutable=['lip'])
    y_2, mutated_2 = model.apply(params, batch_2, mutable=['lip'])

    L = jnp.abs(y_1 - y_2) / jnp.linalg.norm(batch_1 - batch_2)

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
  jax.config.update("jax_enable_x64", False)
  absltest.main()
