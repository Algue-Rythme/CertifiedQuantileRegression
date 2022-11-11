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
from cnqr.layers import StiefelDense, groupsort2
from cnqr._src.lipschitz import power_iteration, spectral_normalization
from cnqr._src.lipschitz import l2_normalize, bjorck_projection, projection_stiefel_manifold
from cnqr._src.lipschitz import implicit_differentiation_stiefel_proj


class StiefelDenseTest(jtu.JaxTestCase):

  @parameterized.parameters(((20, 10),), ((30, 30),), ((10, 60),))
  def test_power_iteration(self, kernel_shape):
    """Test power iteration forward."""
    onp.random.seed(6813)

    kernel = onp.random.normal(size=kernel_shape)
    u_init = onp.random.normal(size=(kernel_shape[-1],))
    u = power_iteration(kernel, u_init, unroll=False, jit=True)
    v = jnp.dot(kernel, u)
    rayleigh = jnp.dot(v, jnp.dot(kernel, u))
    sigma = rayleigh ** 0.5

    sigma_scipy = norm(kernel, ord=2)

    atol = 1e-4
    rtol = 1e-4
    self.assertAllClose(norm(u, ord=2), 1., atol=atol, rtol=rtol)
    self.assertAllClose(norm(v, ord=2), sigma, atol=atol, rtol=rtol)
    self.assertAllClose(sigma, sigma_scipy, atol=atol, rtol=rtol)

  @parameterized.parameters(((6, 1),), ((6, 6),), ((1, 6),))
  def test_power_iteration_backward(self, kernel_shape):
    """Test power iteration VJP."""
    onp.random.seed(1324)

    kernel = onp.random.normal(size=kernel_shape)
    u_init = onp.random.normal(size=(kernel_shape[-1],))

    def run_power_iteration(kernel):
      u = power_iteration(kernel, u_init, unroll=True, jit=True)
      v = jnp.dot(kernel, u)
      rayleigh = jnp.dot(v, jnp.dot(kernel, u))
      sigma = rayleigh ** 0.5
      return sigma

    eps = 1e-3
    rtol = 1e-4
    jtu.check_grads(run_power_iteration, (kernel,), order=1, eps=eps, rtol=rtol)

  @parameterized.parameters(((20, 10),), ((30, 30),), ((10, 60),))
  def test_spectral_normalization(self, kernel_shape):
    """Test spectral normalization forward."""
    onp.random.seed(967)

    kernel = onp.random.normal(size=kernel_shape)
    u_init = onp.random.normal(size=(kernel_shape[-1],))
    ortho, u, sigma = spectral_normalization(kernel, u_init, unroll=False, jit=True)

    K_hat_u = jnp.matmul(ortho, u)
    K_u = jnp.matmul(kernel, u)

    atol = 1e-4
    rtol = 1e-4
    self.assertAllClose(K_hat_u * sigma, K_u, atol=atol, rtol=rtol)

    # check that the matrix is unitary and have the same direction as the original matrix.
    self.assertAllClose(ortho, kernel / norm(kernel, ord=2), atol=atol, rtol=rtol)

  @parameterized.parameters(((6, 1),), ((6, 6),), ((1, 6),))
  def test_spectral_normalization_backward(self, kernel_shape):
    """Test spectral normalization VJP."""
    onp.random.seed(139)

    kernel = onp.random.normal(size=kernel_shape)
    u_init = onp.random.normal(size=(kernel_shape[-1],))
    cotangeant = onp.random.normal(size=kernel_shape)

    def run_spectral_normalization(kernel):
      normalized, _, _ = spectral_normalization(kernel, u_init, unroll=True, jit=True)
      loss = jnp.sum(normalized * cotangeant)
      return loss
    
    eps = 1e-3
    rtol = 1e-4
    jtu.check_grads(run_spectral_normalization, (kernel,), order=1, eps=eps, rtol=rtol)

  @parameterized.parameters(((20, 10),), ((30, 30),), ((10, 60),))
  def test_projection_stiefel_manifold(self, kernel_shape):
    """Test projection onto stiefel manifold of arbitrary matrices."""
    onp.random.seed(9874)

    kernel = onp.random.normal(size=kernel_shape)
    u_init = onp.random.normal(size=(kernel_shape[-1],))

    K, u, sigma = projection_stiefel_manifold(kernel, u_init)
    if kernel_shape[0] > kernel_shape[1]:
      gram_K = jnp.matmul(K.T, K)
    else:
      gram_K = jnp.matmul(K, K.T)

    atol = 1e-4
    rtol = 1e-4
    self.assertAllClose(gram_K, jnp.eye(*gram_K.shape), atol=atol, rtol=rtol)

  @parameterized.parameters(((6, 1),), ((6, 6),), ((1, 6),))
  def test_projection_stiefel_manifold_unrolled(self, kernel_shape):
    """Test projection onto stiefel manifold of arbitrary matrices."""
    onp.random.seed(198)

    kernel = onp.random.normal(size=kernel_shape)
    u_init = onp.random.normal(size=(kernel_shape[-1],))
    cotangeant = onp.random.normal(size=kernel_shape)

    def run_stiefel_projection_unrolled(kernel):
      theta, _, _ = projection_stiefel_manifold(kernel, u_init, unroll=True, jit=True)
      loss = jnp.sum(theta * cotangeant)
      return loss

    eps = 1e-3
    rtol = 1e-4
    jtu.check_grads(run_stiefel_projection_unrolled, (kernel,), order=1, eps=eps, rtol=rtol)

  def test_projection_stiefel_manifold_implicit(self):
    """Test projection onto stiefel manifold of arbitrary matrices."""
    onp.random.seed(937)

    kernel_shape = (10, 10)
    kernel = onp.random.normal(size=kernel_shape)
    u_init = onp.random.normal(size=(kernel_shape[-1],))
    cotangeant = onp.random.normal(size=kernel_shape)

    def run_stiefel_projection_implicit(kernel):
      proj = partial(projection_stiefel_manifold, u_init=u_init, unroll=False, jit=True)
      proj_implicit = implicit_differentiation_stiefel_proj(proj, solver='eigen', has_aux=True)
      theta, _, _ = proj_implicit(kernel)
      loss = jnp.sum(theta * cotangeant)
      return loss

    eps = 1e-3
    rtol = 1e-4
    jtu.check_grads(run_stiefel_projection_implicit, (kernel,), modes=['rev'], order=1, eps=eps, rtol=rtol)

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

    self.assertAllClose(mutated_1['lip']['u'], mutated_2['lip']['u'], atol=atol, rtol=rtol)

  def test_stiefel_dense_unrolled(self):
    key = jax.random.PRNGKey(444)
    key_params, key_lip, batch_key = jax.random.split(key, 3)

    batch_size = 16
    input_size = 5
    batch = jax.random.normal(key=batch_key, shape=(batch_size, input_size))

    features = 15
    model = StiefelDense(features=features, implicit_diff=False)
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
    model = StiefelDense(features=features, implicit_diff=True)
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

    batch_size = 1
    input_size = 10
    batch_1 = jax.random.normal(key=batch_1_key, shape=(batch_size, input_size))
    batch_2 = jax.random.normal(key=batch_2_key, shape=(batch_size, input_size))

    features = [32, 48, 4, 1]
    stiefel_dense = partial(StiefelDense, train=True)
    model = nn.Sequential([stiefel_dense(num_features) for num_features in features])
    params = model.init(rngs={'params': key_params, 'lip': key_lip}, inputs=batch_1)

    y_1, mutated_1 = model.apply(params, batch_1, mutable=['lip'])
    y_2, mutated_2 = model.apply(params, batch_2, mutable=['lip'])

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
