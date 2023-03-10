# This file is part of CNQR which is released under the Apache License 2.0.
# See file LICENSE in the root directory or https://opensource.org/licenses/Apache-2.0 for full license details.

"""Tests for PICNN network architecture."""
from absl.testing import parameterized
from absl.testing import absltest

import jax
import jax.numpy as jnp
import jax._src.test_util as jtu
import flax.linen as nn

from cnqr.layers import PICNN


class PICNNTest(jtu.JaxTestCase):

    @parameterized.parameters((10, 2, 4, (64, 64), 1), (10, 5, 5, (64, 64), 0.01), (10, 4, 2, (64, 64, 64), 0.1))
    def test_picnn_convexity(self, n_samples, n_features_x, n_features_y, dim_hidden, epsilon_init):
        """Tests convexity of PICNN (wrt y) with identity initialization and with random
          initialization (higher values of epislon)."""
        rng = jax.random.PRNGKey(999)
        # define picnn model
        picnn = PICNN(dim_hidden, dim_y=n_features_y, epsilon_init=epsilon_init)

        # initialize model
        key1, key2, key3 = jax.random.split(rng, 3)
        params = picnn.init(key1, x=jnp.ones((1, n_features_x)), y=jnp.ones((1, n_features_y)))['params']

        # check convexity
        x = jax.random.normal(key1, (n_samples, n_features_x))
        y1 = jax.random.normal(key2, (n_samples, n_features_y))
        y2 = jax.random.normal(key3, (n_samples, n_features_y))

        out_y1 = picnn.apply({'params': params}, x=x, y=y1)
        out_y2 = picnn.apply({'params': params}, x=x, y=y2)

        out = list()
        for t in jnp.linspace(0, 1, num=25):
            out_interp = picnn.apply({'params': params}, x=x, y=t * y1 + (1 - t) * y2)
            out.append((t * out_y1 + (1 - t) * out_y2) - out_interp)
        tol = 1e-5
        self.assertTrue(jnp.all(jnp.stack(out) + tol >= 0))

    @parameterized.parameters((10, 2, 4, (64, 64), 1), (10, 5, 5, (64, 64), 0.01), (10, 4, 2, (64, 64, 64), 0.1))
    def test_picnn_hessian(self, n_samples, n_features_x, n_features_y, dim_hidden, epsilon_init):
        """Tests if Hessian of PICNN (wrt y) is positive-semidefinite with identity initialization and with
          random initialization (higher values of epislon)."""
        rng = jax.random.PRNGKey(999)
        # define picnn model
        picnn = PICNN(dim_hidden, dim_y=n_features_y, epsilon_init=epsilon_init)

        # initialize model
        key1, key2, key3 = jax.random.split(rng, 3)
        params = picnn.init(key1, x=jnp.ones((1, n_features_x)), y=jnp.ones((1, n_features_y)))['params']

        # check if Hessian is positive-semidefinite via eigenvalues
        x_data = jax.random.normal(key2, (n_samples, n_features_x))
        y_data = jax.random.normal(key3, (n_samples, n_features_y))

        # compute Hessian
        W = []
        for x_i, y_i in zip(x_data, y_data):
            hessian = jax.jacfwd(jax.jacrev(picnn.apply, argnums=2), argnums=2)  # wrt y
            picnn_hess = hessian({'params': params}, x_i[jnp.newaxis, ...], y_i[jnp.newaxis, ...])\
                .reshape((n_features_y, n_features_y))
            # compute eigenvalues
            w, _ = jnp.linalg.eig(picnn_hess)
            W.append(w)
        tol = 1e-5
        self.assertTrue(jnp.all(jnp.stack(W) + tol >= 0))

    @parameterized.parameters((10, 2, 4, (64, 64)), (10, 5, 5, (64, 64)), (10, 4, 2, (64, 64, 64)))
    def test_picnn_grad_identity_at_initialization(self, n_samples, n_features_x, n_features_y, dim_hidden):
        """Tests if the gradient of a PICNN (wrt y) is equivalent to the identity function at initialization"""
        rng = jax.random.PRNGKey(999)
        # define picnn model
        picnn = PICNN(dim_hidden, dim_y=n_features_y, epsilon_init=1e-5,
                      tau_act_fn=nn.softplus, sigma_act_fn=nn.relu, pos_act_fn=nn.relu)

        # initialize model
        key1, key2, key3 = jax.random.split(rng, 3)
        params = picnn.init(key1, x=jnp.ones((1, n_features_x)), y=jnp.ones((1, n_features_y)))['params']

        # check if transport function is initially close to identity function
        x_data = jax.random.normal(key2, (n_samples, n_features_x))
        y_data = jax.random.normal(key3, (n_samples, n_features_y))

        # compute grads
        batch_forward = lambda x, y: picnn.apply({'params': params}, x[jnp.newaxis, ...], y[jnp.newaxis, ...])[0]
        grads = jax.vmap(jax.grad(batch_forward, argnums=1), in_axes=(0, 0))(x_data, y_data)
        rtol, atol = 1e-3, 1e-4
        self.assertAllClose(y_data, grads, atol=atol, rtol=rtol)


def picnn_summary(n_features_x, n_features_y, dim_hidden):
    """Summary of PICNN module"""
    picnn = PICNN(dim_hidden, dim_y=n_features_y)
    print(picnn.tabulate(jax.random.PRNGKey(0), jnp.ones((1, n_features_x)), jnp.ones((1, n_features_y))))


if __name__ == '__main__':
    picnn_summary(2, 2, (64, 64))
    jax.config.update("jax_enable_x64", False)
    absltest.main()
