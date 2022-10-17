# This file is part of CNQR which is released under the Apache License 2.0.
# See file LICENSE in the root directory or https://opensource.org/licenses/Apache-2.0 for full license details.

"""Computation of Wasserstein-2 distance in Jax/Flax with CNQR layers on Two Moons.

It is based on Kantorovich-Rubinstein (KR) duality with convex conjugate functions.

Follow guidelines from [1].

[1] Makkuva, A., Taghvaei, A., Oh, S. and Lee, J., 2020, November.
    Optimal transport mapping via input convex neural networks.
    In International Conference on Machine Learning (pp. 6672-6681). PMLR.
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Sequence, Tuple

from absl import app
from absl import flags
from absl import logging

import jax
import jax.numpy as jnp
from jax.config import config

from jaxopt.tree_util import tree_negative

from flax.training import train_state
from flax import linen as nn
from flax import core

import optax
from sklearn.datasets import make_moons, make_circles

from cnqr.losses import symmetric_KR_W2
from cnqr.layers import StiefelDense, full_sort, groupsort2
from cnqr.layers import ICNN


flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
flags.DEFINE_float("noise", 0.05, "Noise in observations.")
flags.DEFINE_integer("dataset_size", 5000, "Number of points in dataset.")
flags.DEFINE_integer("num_epochs", 100, "Number of passes over the dataset.")
flags.DEFINE_enum("dataset", "two_moons", ['two_moons', 'two_circles'], "Dataset to train on.")
flags.DEFINE_integer("batch_size", 256, "Number of examples in batch.")
flags.DEFINE_bool("relax_g_convexity", True, "Relax convexity condition of g potential.")
flags.DEFINE_float("inv_beta", 10., "Inverse strength of regularization for g convexity. When inv_beta->0, g is convex.")
FLAGS = flags.FLAGS


class DualPotentials(nn.Module):
  """A pair of ICNNs for computation of Wasserstein-2 distance.
  
  Attributes:
      dim_hidden: Dimension of hidden layers.

  Methods:
      f: architecture for X~P Must be a convex function.
      g: architecture for Y~Q. Strict convexity is not required.
  """
  dim_hidden: Sequence[int] = (64, 64, 64, 64)
  f_architecture: nn.Module = ICNN
  g_architecture: nn.Module = ICNN

  def setup(self):
    self.f = self.f_architecture(name='f', dim_hidden=self.dim_hidden)
    self.g = self.g_architecture(name='g', dim_hidden=self.dim_hidden,
                                  relax_strict_convexity=FLAGS.relax_g_convexity)

  def approx_f_convex_conjugate(self, y):
    """Approximates f^*(y) = min_x <x, y> - f(x).
    
      x^* = argmin_x <x, y> - f(x)

    It is assumed that x^* = nabla_y g(y).

    Args:
      y: y~Q.

    Returns:
      f^*(y) = min_x <x, y> - f(x) = <x^*, y> - f(x^*).
    """

    def grad_g(y):
      single_y = y[jnp.newaxis, ...]  # fake batch dimension.
      scalar_g = lambda y: jnp.squeeze(self.g(y))
      grad_single_y = jax.grad(scalar_g)(single_y)
      grad_y = jnp.reshape(grad_single_y, y.shape)
      return grad_y
    
    x_star = jax.vmap(grad_g, in_axes=0, out_axes=0)(y)

    xy = jax.vmap(jnp.vdot, in_axes=(0, 0))(x_star, y)
    fx = self.f(x_star)

    return xy - fx

  def __call__(self, x, y):
    fx = self.f(x)
    f_conj_y = self.approx_f_convex_conjugate(y)
    return fx, f_conj_y


def compatible_tabulate(model, keys, dummy_batch):
  # TODO: remove this hack once Flax's API is mature and settle definetely for a recent version.
  try:
    infos = model.tabulate(keys, dummy_batch,
                           exclude_methods=('approx_f_convex_conjugate',))
  except AttributeError:
    # `exclude_methods`` is not supported in newer versions of Flax (>0.6.0).
    # But it is mandatory before when the layer has multiple methods.
    infos = model.tabulate(keys, dummy_batch)
  finally:
    logging.info(infos)


def create_train_state(rng):
  """Creates initial `TrainState`."""
  model = DualPotentials()
  dummy_batch = jnp.zeros([FLAGS.batch_size, 2])
  params = model.init(rng, dummy_batch, dummy_batch)
  # compatible_tabulate(model, rng, dummy_batch)
  tx = optax.adam(FLAGS.learning_rate)
  return train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=tx)

def convex_regularization(g_params):
  penalty = 0
  for k, param in g_params.items():
    is_pos = k.startswith("w_U_dense_") and "kernel" in param
    if is_pos:
      negative = jax.nn.relu(-param["kernel"])
      cost = jnp.sum(negative ** 2)
      penalty += 0.5 * cost
  return penalty / FLAGS.inv_beta

@jax.jit
def apply_model(state, P, Q):
  """Computes gradients and loss for a single batch."""

  def balanced_wasserstein(params):
    fP, f_star_Q = state.apply_fn(params, P, Q)
    loss = symmetric_KR_W2(fP, f_star_Q)
    if FLAGS.relax_g_convexity:
      Rg = convex_regularization(params["params"]["g"])
      loss = loss + Rg
    return loss, (fP, f_star_Q)

  grad_fn = jax.value_and_grad(balanced_wasserstein, has_aux=True)
  (loss, aux), grads = grad_fn(state.params)

  grads = core.unfreeze(grads)
  grads["params"]["f"] = tree_negative(grads["params"]["f"])  # f is a argmax, not a argmin.
  grads = core.freeze(grads)

  return grads, loss, aux

@jax.jit
def approx_w2(P, Q, fP, f_star_Q):
  """Compute approximation of Wasserstein-2 distance from (f,g) predictions."""
  cost = -jnp.mean(fP) - jnp.mean(f_star_Q)
  squared_norm = lambda x: jnp.sum(x**2)
  norms = jax.vmap(squared_norm, in_axes=0)
  # C_pq = 0.5 * \|X\|^2_2 + 0.5 * \|Y\|^2_2
  C_pq = 0.5 * (jnp.mean(norms(P)) + jnp.mean(norms(Q)))
  cost = cost + C_pq
  return cost

def train_epoch(state, train_ds, batch_size, rng):
  """Train for a single epoch."""
  train_ds_size = len(train_ds["P"])
  steps_per_epoch = train_ds_size // batch_size

  perms = jax.random.permutation(rng, train_ds_size)
  perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))

  epoch_loss = []
  w2_approx = []

  for step, perm in enumerate(perms):
    P = train_ds["P"][perm, ...]
    Q = train_ds["Q"][perm, ...]
    grads, loss, (fP, f_star_Q) = apply_model(state, P, Q)
    state = state.apply_gradients(grads=grads)
    w2 = approx_w2(P, Q, fP, f_star_Q)
    epoch_loss.append(loss)
    w2_approx.append(w2)

  train_loss = -jnp.mean(jnp.array(epoch_loss))
  w2_approx = jnp.mean(jnp.array(w2_approx))
  return state, (train_loss, w2_approx)


def get_datasets():
  if FLAGS.dataset == 'two_moons':
    make_ds = make_moons
  elif FLAGS.dataset == 'two_circles':
    make_ds = make_circles
  n_samples = FLAGS.dataset_size // 2
  X, y = make_ds(n_samples=(n_samples, n_samples), noise=FLAGS.noise)
  X = X.astype(jnp.float32)
  X = (X - jnp.mean(X, axis=0)) / jnp.std(X, axis=0)  # centered
  Q = X[y == 0, :]
  P = X[y == 1, :]
  train_ds = {'P': P, 'Q': Q}
  return train_ds


def compute_wasserstein_2_distance(P, Q, epsilon):
  import ott
  from ott.geometry import pointcloud
  from ott.core import sinkhorn
  from ott.tools import transport
  print("Compute Wasserstein-2...")
  geom = pointcloud.PointCloud(P, Q, epsilon=epsilon, power=2.0)
  out = sinkhorn.sinkhorn(geom)
  # The 0.5 factor in literature is not taken into account in OTT.
  # This introduces a factor 0.5 error in entropic regularization.
  # Results should be more accurate with smaller epsilon.
  reg_ot_cost = out.reg_ot_cost * 0.5
  return reg_ot_cost


def plot_data(P, Q, train_state):
  import matplotlib.pyplot as plt
  from matplotlib import cm
  import numpy as onp

  print("Plot OT plan...")

  shift = 0.5
  points = jnp.concatenate([P, Q], axis=0)
  x = onp.linspace(points[:, 0].min() - shift, points[:, 0].max() + shift, 120)
  y = onp.linspace(points[:, 1].min() - shift, points[:, 1].max() + shift, 120)
  xx, yy = onp.meshgrid(x, y, sparse=False)
  X_pred = onp.stack((xx.ravel(), yy.ravel()), axis=1)
  X_pred = jnp.array(X_pred)

  fP, f_star_Q = train_state.apply_fn(train_state.params, X_pred, X_pred)

  # plot.
  fig = plt.figure(figsize=(10, 7))
  ax_0 = fig.add_subplot(211)
  ax_1 = fig.add_subplot(212)
  for ax, preds, name in zip((ax_0, ax_1), (fP, f_star_Q), ('f(P)', 'f*(Q)')):
    preds = preds.reshape(x.shape[0], y.shape[0])
    ax.scatter(P[:, 0], Q[:, 1], alpha=0.1)
    ax.scatter(P[:, 0], Q[:, 1], alpha=0.1)
    cset = ax.contour(xx, yy, preds, cmap='twilight', levels=20)
    ax.clabel(cset, inline=1, fontsize=10)
    ax.set_title(name)

  file_loc = 'sandbox/wasserstein_2.png'
  plt.savefig(file_loc)
  print(f"OT plan saved at {file_loc}.")


def main(argv):
  train_ds = get_datasets()
  rng = jax.random.PRNGKey(0)

  rng, init_rng = jax.random.split(rng)
  state = create_train_state(init_rng)

  for epoch in range(1, FLAGS.num_epochs + 1):
    rng, input_rng = jax.random.split(rng)
    state, metrics = train_epoch(state, train_ds,
                                 FLAGS.batch_size,
                                 input_rng)
    train_loss, w2 = metrics
    logging.info(f'epoch:[{epoch:3d}/{FLAGS.num_epochs}], train_loss: {train_loss:.4f}, W2: {w2:.4f}')

  epsilon = 1e-2
  sinkhorn_w2 = compute_wasserstein_2_distance(train_ds['P'], train_ds['Q'], epsilon)
  print(f"Wasserstein-2: {w2:.2f}")
  print(f"Regulized Wasserstein-2 (ε={epsilon:.2f}): {float(sinkhorn_w2):.2f}")
  error = jnp.abs(sinkhorn_w2 - w2) / w2 * 100
  print(f"Relative Error: {error:.2f}%")

  plot_data(train_ds['P'], train_ds['Q'], state)

if __name__ == '__main__':
  # jax.config.update("jax_enable_x64", True)
  app.run(main)
