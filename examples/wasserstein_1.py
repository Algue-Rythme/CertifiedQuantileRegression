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

from flax.training import train_state
from flax import linen as nn

import optax
from sklearn.datasets import make_moons, make_circles

from cnqr.layers import BjorckDense, groupsort2


flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
flags.DEFINE_float("noise", 0.05, "Noise in observations.")
flags.DEFINE_integer("dataset_size", 5000, "Number of points in dataset.")
flags.DEFINE_integer("num_epochs", 30, "Number of passes over the dataset.")
flags.DEFINE_enum("dataset", "two_moons", ['two_moons', 'two_circles'], "Dataset to train on.")
flags.DEFINE_integer("batch_size", 50, "Number of examples in batch.")
flags.DEFINE_integer("maxiter_spectral", 3, "Number of iterations for spectral norm.")
flags.DEFINE_integer("maxiter_bjorck", 15, "Number of iterations for Bjorck.")
FLAGS = flags.FLAGS


class LipschitzNN(nn.Module):
    hidden_widths: Sequence[int] = (64, 64, 64)

    @nn.compact
    def __call__(self, inputs, train):
        bjorck_dense = partial(BjorckDense,
            maxiter_spectral=FLAGS.maxiter_spectral,
            maxiter_bjorck=FLAGS.maxiter_bjorck,
            train=train)

        x = inputs
        for width in self.hidden_widths:
            x = bjorck_dense(features=width)(x)
            x = groupsort2(x)
        x = bjorck_dense(features=1)(x)

        return x


class LipschitzTrainState(train_state.TrainState):
    """Train state with Lipschitz constraint."""
    lip_state: Any

def create_train_state(rng):
    """Creates initial `TrainState`."""
    model = LipschitzNN()
    keys = dict(zip(['params', 'lip'], jax.random.split(rng, 2)))
    dummy_batch = jnp.zeros([FLAGS.batch_size, 2])
    model_params = model.init(keys, dummy_batch, train=True)
    logging.info(model.tabulate(keys, dummy_batch, train=True))
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

    def loss_fn(params):
        model_params = {'params': params, 'lip':state.lip_state}
        score, variables = state.apply_fn(model_params, points, train=True, mutable='lip')
        P = score * (labels == 1).astype(jnp.float32)
        Q = score * (labels == -1).astype(jnp.float32)
        loss = -(jnp.mean(P) - jnp.mean(Q))  # maximize E_Pf - E_Qf
        return loss, variables['lip']

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, lip_vars), grads = grad_fn(state.params)
    return grads, lip_vars, loss

@jax.jit
def update_model(state, grads, lip_vars):
    return state.apply_gradients(grads=grads, lip_state=lip_vars)


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
        grads, lip_vars, loss = apply_model(state, points, labels)
        state = update_model(state, grads, lip_vars)
        epoch_loss.append(loss)

    train_loss = -jnp.mean(jnp.array(epoch_loss))
    return state, train_loss


def get_datasets():
    if FLAGS.dataset == 'two_moons':
        make_ds = make_moons
    elif FLAGS.dataset == 'two_circles':
        make_ds = make_circles
    X, y = make_ds(n_samples=FLAGS.dataset_size, noise=FLAGS.noise)
    X = X.astype(jnp.float32)
    y = y.astype(jnp.float32)
    X = (X - jnp.mean(X, axis=0)) / jnp.std(X, axis=0)
    y = 2*y - 1  # label set is now {-1, 1}
    train_ds = {'points':X, 'labels':y}
    return train_ds


def main(argv):
    train_ds = get_datasets()
    rng = jax.random.PRNGKey(0)

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng)

    for epoch in range(1, FLAGS.num_epochs + 1):
        rng, input_rng = jax.random.split(rng)
        state, train_loss = train_epoch(state, train_ds,
                                        FLAGS.batch_size,
                                        input_rng)
        logging.info(f'epoch:{epoch:3d}, train_loss: {train_loss:.4f}')


if __name__ == '__main__':
    app.run(main)
