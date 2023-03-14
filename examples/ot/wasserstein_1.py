# This file is part of CNQR which is released under the Apache License 2.0.
# See file LICENSE in the root directory or https://opensource.org/licenses/Apache-2.0 for full license details.

"""Computation of Wasserstein-1 distance in Jax/Flax with CNQR layers on Two Moons.

It is based on Kantorovich-Rubinstein (KR) duality.
"""

from functools import partial
from typing import Any, Callable, Sequence, Tuple

from absl import app
from absl import flags
from absl import logging

import jax
import jax.numpy as jnp
from jax.config import config

from flax.training import train_state
from flax import linen as nn
from matplotlib import axes
import numpy as onp

import optax
from sklearn.datasets import make_moons, make_circles

from cnqr.losses import balanced_KR
from cnqr.layers import StiefelDense, Normalized2ToInftyDense, fullsort, groupsort2


flags.DEFINE_float("learning_rate", 1e-2, "Learning rate.")
flags.DEFINE_float("noise", 0.05, "Noise in observations.")
flags.DEFINE_integer("dataset_size", 4096, "Number of points in dataset.")
flags.DEFINE_integer("num_epochs", 200, "Number of passes over the dataset.")
flags.DEFINE_enum("dataset", "two_moons", ['two_moons', 'two_circles'], "Dataset to train on.")
flags.DEFINE_integer("batch_size", 256, "Number of examples in batch.")
FLAGS = flags.FLAGS


class LipschitzNN(nn.Module):
  hidden_widths: Sequence[int] = (256, 128, 64)

  @nn.compact
  def __call__(self, inputs, train):

    x = inputs
    for width in self.hidden_widths:
      x = StiefelDense(features=width)(x, train=train)  
      x = fullsort(x)
    x = Normalized2ToInftyDense(features=1, use_bias=False)(x, train=train)

    return x


def compatible_tabulate(model, keys, dummy_batch):
  # TODO: remove this hack once Flax's API is mature and settle definetely for a recent version.
  try:
    infos = model.tabulate(keys, dummy_batch, train=True,
                           exclude_methods=('_stiefel_projection',))
  except AttributeError:
    # `exclude_methods`` is not supported in newer versions of Flax (>0.6.0).
    # But it is mandatory before when the layer has multiple methods.
    infos = model.tabulate(keys, dummy_batch, train=True)
  finally:
    logging.info(infos)


class LipschitzTrainState(train_state.TrainState):
  """Train state with Lipschitz constraint."""
  lip_state: Any

def create_train_state(rng):
  """Creates initial `TrainState`."""
  model = LipschitzNN()
  keys = dict(zip(['params', 'lip'], jax.random.split(rng, 2)))
  dummy_batch = jnp.zeros([FLAGS.batch_size, 2])
  model_params = model.init(keys, dummy_batch, train=True)
  # compatible_tabulate(model, keys, dummy_batch)
  params, lip_state = model_params['params'], model_params['lip']
  tx = optax.adam(FLAGS.learning_rate)
  return LipschitzTrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=tx,
    lip_state=lip_state)

@jax.jit
def apply_model(state, points, labels):
  """Computes gradients, loss and accuracy for a single batch."""

  def balanced_wasserstein(params):
    model_params = {'params': params, 'lip': state.lip_state}
    score, variables = state.apply_fn(model_params, points, train=True, mutable='lip')
    score = score.flatten()  # size (B,)
    loss, (pred_P, pred_Q) = balanced_KR(score, labels, has_aux=True)
    return loss, (variables['lip'], pred_P, pred_Q)

  grad_fn = jax.value_and_grad(balanced_wasserstein, has_aux=True)
  (loss, aux), grads = grad_fn(state.params)
  return grads, aux, loss

@jax.jit
def update_model(state, grads, lip_vars):
  return state.apply_gradients(grads=grads, lip_state=lip_vars)


def predict_model(train_state, points):
  model_params = {'params': train_state.params, 'lip': train_state.lip_state}
  score = train_state.apply_fn(model_params, points, train=False)
  return score


def train_epoch(state, train_ds, batch_size, rng):
  """Train for a single epoch."""
  train_ds_size = len(train_ds["points"])
  steps_per_epoch = train_ds_size // batch_size

  perms = jax.random.permutation(rng, train_ds_size)
  perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))

  epoch_loss = []

  for step, perm in enumerate(perms):
    points = train_ds["points"][perm, ...]
    labels = train_ds["labels"][perm, ...]
    grads, aux, loss = apply_model(state, points, labels)
    (lip_vars, P, Q) = aux
    state = update_model(state, grads, lip_vars)
    epoch_loss.append(loss)

  train_loss = -jnp.mean(jnp.array(epoch_loss))
  return state, train_loss, (P.sum(), Q.sum())


def get_datasets():
  if FLAGS.dataset == 'two_moons':
    make_ds = make_moons
  elif FLAGS.dataset == 'two_circles':
    make_ds = make_circles
  X, y = make_ds(n_samples=FLAGS.dataset_size, noise=FLAGS.noise)
  X = X.astype(jnp.float32)
  y = y.astype(jnp.float32)
  X = (X - jnp.mean(X, axis=0)) / jnp.std(X, axis=0)  # centered
  y = 2 * y - 1  # label set is now {-1, 1}
  train_ds = {'points': X, 'labels': y}
  return train_ds


def compute_earth_mover_distance_ott(points, labels, epsilon=1e-2):
  """Compute the Earth Mover's Distance (EMD) between two distributions.

  Based on the OTT library: https://ott-jax.readthedocs.io
  """
  import ott
  from ott.geometry import costs, pointcloud
  from ott.problems.linear import linear_problem
  from ott.solvers.linear import sinkhorn
  print("Compute EMD...")
  p, q = points[labels == 1], points[labels == -1]
  geom = pointcloud.PointCloud(p, q, epsilon=epsilon,
                               cost_fn=costs.Euclidean)
  ot_prob = linear_problem.LinearProblem(geom)
  solver = sinkhorn.Sinkhorn()
  out = solver(ot_prob)
  emd = out.reg_ot_cost
  print(f"Regulized EMD (ε={epsilon:.2f}): {float(emd):.2f}")
  return emd


def compute_earth_mover_distance_pot(points, labels):
  """Compute the Earth Mover's Distance (EMD) between two distributions.
  
  Based on the POT library: https://pythonot.github.io/"""
  import ot
  print("Compute EMD...")
  p, q = points[labels == 1], points[labels == -1]
  subset_size = 1024
  p = onp.random.permutation(p)[:subset_size]
  q = onp.random.permutation(q)[:subset_size]
  # uniform distribution on samples
  M = ot.dist(p, q, metric='euclidean')
  emd = ot.emd2([], [], M)
  print(f"EMD (LinearSolving): {float(emd):.2f}")
  return emd 


def plot_data(points, labels, train_state):
  import matplotlib.pyplot as plt
  from matplotlib import cm
  import numpy as onp

  print("Plot OT plan...")

  # forward in f
  shift = 0.5
  x = onp.linspace(points[:, 0].min() - shift, points[:, 0].max() + shift, 120)
  y = onp.linspace(points[:, 1].min() - shift, points[:, 1].max() + shift, 120)
  xx, yy = onp.meshgrid(x, y, sparse=False)
  X_pred = onp.stack((xx.ravel(), yy.ravel()), axis=1)

  # make predictions of f
  Y_pred = predict_model(train_state, jnp.array(X_pred))
  Y_pred_square = Y_pred.reshape(x.shape[0], y.shape[0])

  fP = predict_model(train_state, jnp.array(points[labels == 1])).flatten()
  fQ = predict_model(train_state, jnp.array(points[labels == -1])).flatten()

  # plot contours of f
  fig = plt.figure(figsize=(10, 7))
  ax = fig.add_subplot(121,aspect='equal')
  ax.scatter(points[labels == 1, 0], points[labels == 1, 1], alpha=0.1)
  ax.scatter(points[labels == -1, 0], points[labels == -1, 1], alpha=0.1)
  cset = ax.contour(xx, yy, Y_pred_square, cmap='twilight', levels=20)
  ax.clabel(cset, inline=1, fontsize=10)

  # Plot histogram of fP and fQ
  ax = fig.add_subplot(122)
  ax.hist(fP, alpha=0.5, label='1')
  ax.hist(fQ, alpha=0.5, label='-1')

  file_loc = 'sandbox/wasserstein_1.png'
  plt.savefig(file_loc)
  print(f"OT plan saved at {file_loc}.")


def main(argv):
  train_ds = get_datasets()
  rng = jax.random.PRNGKey(37)

  rng, init_rng = jax.random.split(rng)
  state = create_train_state(init_rng)

  for epoch in range(1, FLAGS.num_epochs + 1):
    rng, input_rng = jax.random.split(rng)
    state, train_loss, logs = train_epoch(state, train_ds,
                                          FLAGS.batch_size,
                                          input_rng)
    logging.info(f'epoch:[{epoch:3d}/{FLAGS.num_epochs}], train_loss: {train_loss:.4f}, P: {logs[0]:.4f}, Q: {logs[1]:.4f}')

  print(f"Wasserstein-1: {train_loss:.2f}")
  emd = compute_earth_mover_distance_pot(train_ds['points'], train_ds['labels'])
  error = jnp.abs(emd - train_loss) / train_loss * 100
  print(f"Relative Error: {error:.2f}%")

  plot_data(train_ds['points'], train_ds['labels'], state)


if __name__ == '__main__':
  # jax.config.update("jax_enable_x64", True)
  app.run(main)
