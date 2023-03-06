from functools import partial

from absl.testing import parameterized
from absl.testing import absltest

import jax
import jax.numpy as jnp
import jax._src.test_util as jtu
import flax.linen as nn
import numpy as onp

from scipy.linalg import norm, eig

from cnqr._src.parametrizations import power_iteration, spectral_normalization
from cnqr._src.parametrizations import projection_spectral_bjorck
from cnqr._src.parametrizations import implicit_differentiation_stiefel_proj
from cnqr.parametrizations import BjorckParametrization


class BjorckTest(jtu.JaxTestCase):

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
  def test_projection_spectral_bjorck(self, kernel_shape):
    """Test projection onto stiefel manifold of arbitrary matrices."""
    onp.random.seed(9874)

    kernel = onp.random.normal(size=kernel_shape)
    u_init = onp.random.normal(size=(kernel_shape[-1],))

    K, u, sigma = projection_spectral_bjorck(kernel, u_init)
    if kernel_shape[0] > kernel_shape[1]:
      gram_K = jnp.matmul(K.T, K)
    else:
      gram_K = jnp.matmul(K, K.T)

    atol = 1e-4
    rtol = 1e-4
    self.assertAllClose(gram_K, jnp.eye(*gram_K.shape), atol=atol, rtol=rtol)

  @parameterized.parameters(((6, 1),), ((6, 6),), ((1, 6),))
  def test_projection_spectral_bjorck_unrolled(self, kernel_shape):
    """Test projection onto stiefel manifold of arbitrary matrices."""
    onp.random.seed(198)

    kernel = onp.random.normal(size=kernel_shape)
    u_init = onp.random.normal(size=(kernel_shape[-1],))
    cotangeant = onp.random.normal(size=kernel_shape)

    def run_stiefel_projection_unrolled(kernel):
      theta, _, _ = projection_spectral_bjorck(kernel, u_init, unroll=True, jit=True)
      loss = jnp.sum(theta * cotangeant)
      return loss

    eps = 1e-3
    rtol = 1e-4
    jtu.check_grads(run_stiefel_projection_unrolled, (kernel,), order=1, eps=eps, rtol=rtol)

  def test_projection_spectral_bjorck_implicit(self):
    """Test projection onto stiefel manifold of arbitrary matrices."""
    onp.random.seed(937)

    kernel_shape = (10, 10)
    kernel = onp.random.normal(size=kernel_shape)
    u_init = onp.random.normal(size=(kernel_shape[-1],))
    cotangeant = onp.random.normal(size=kernel_shape)

    def run_stiefel_projection_implicit(kernel):
      proj = partial(projection_spectral_bjorck, u_init=u_init, unroll=False, jit=True)
      proj_implicit = implicit_differentiation_stiefel_proj(proj, solver='eigen', has_aux=True)
      theta, _, _ = proj_implicit(kernel)
      loss = jnp.sum(theta * cotangeant)
      return loss

    eps = 1e-3
    rtol = 1e-4
    jtu.check_grads(run_stiefel_projection_implicit, (kernel,), modes=['rev'], order=1, eps=eps, rtol=rtol)

  def test_bjorck_parametrization(self):
    """Test BjorckParametrization."""
    onp.random.seed(123)

    bjorck_param = BjorckParametrization()  # default values.

    kernel_shape = (12, 8)
    kernel = onp.random.normal(size=kernel_shape)

    key_lip = jax.random.PRNGKey(444)
    variables = bjorck_param.init(rngs={'lip': key_lip}, kernel=kernel, train=True)
    assert list(variables.keys()) == ['lip']
    
    # check that ortho_ker is a unitary matrix.
    ortho_ker = bjorck_param.apply(variables=variables, kernel=kernel, train=False)
    gram_ortho_ker = jnp.matmul(ortho_ker.T, ortho_ker)
    self.assertAllClose(gram_ortho_ker, jnp.eye(*gram_ortho_ker.shape))


if __name__ == '__main__':
  jax.config.update("jax_debug_nans", True)
  jax.config.update("jax_enable_x64", False)
  absltest.main()
