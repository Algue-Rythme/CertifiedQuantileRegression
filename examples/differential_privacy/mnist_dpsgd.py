# This file is part of CNQR which is released under the Apache License 2.0.
# See file LICENSE in the root directory or https://opensource.org/licenses/Apache-2.0 for full license details.


"""MNIST example.
Library file which executes the training and evaluation loop for MNIST.
The data is loaded using tensorflow_datasets.
"""

from functools import partial
from typing import Any, Callable, Sequence, Tuple

import warnings

from absl import logging
from absl import app
from absl import flags

import dp_accounting

from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp

import ml_collections

import numpy as np
import optax

import tensorflow as tf
import tensorflow_datasets as tfds

from tqdm import tqdm

from cnqr.losses import multiclass_hinge
from cnqr.layers import StiefelDense, Normalized2ToInftyDense
from cnqr.layers import RKOConv, l2norm_pool, global_l2norm_pool
from cnqr.layers import fullsort, groupsort2


flags.DEFINE_float("learning_rate", 5e-2, "Learning rate.")
flags.DEFINE_integer("num_epochs", 30, "Number of passes over the dataset.")
flags.DEFINE_integer("batch_size", 256, "Number of examples in batch.")
flags.DEFINE_integer("batch_size_test", 1024, "Number of examples in batch at test time.")

flags.DEFINE_boolean("use_global_pool", False, "Use l2norm pooling.")
flags.DEFINE_boolean("use_bias", False, "Use bias in the model.")

flags.DEFINE_string("loss_fn", "multiclass_hinge", "Loss function to use.")
flags.DEFINE_float("temperature", 1., "Temperature for softmax.")
flags.DEFINE_float("margin", 0.05, "Margin for multiclass hinge loss.")

flags.DEFINE_boolean("dpsgd", True, "If True, train with DP-SGD. If False, use Adam.")
flags.DEFINE_float("noise_multiplier", 1.2, "Ratio of the standard deviation to the clipping norm.")
flags.DEFINE_float("l2_norm_clip", 2., "Clipping norm.")
flags.DEFINE_float("delta", 1e-5, "Target delta.")


FLAGS = flags.FLAGS
NUM_EXAMPLES = 60 * 1000


def compute_epsilon(steps, target_delta=1e-5):
  if NUM_EXAMPLES * target_delta > 1.:
    warnings.warn('Your delta might be too high.')
  q = FLAGS.batch_size / float(NUM_EXAMPLES)
  orders = list(jnp.linspace(1.1, 10.9, 99)) + list(range(11, 64))
  accountant = dp_accounting.rdp.RdpAccountant(orders)
  accountant.compose(dp_accounting.PoissonSampledDpEvent(
      q, dp_accounting.GaussianDpEvent(FLAGS.noise_multiplier)), steps)
  return accountant.get_epsilon(target_delta)


def loss_on_batch(logits, labels):
  one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
  if FLAGS.loss_fn == "multiclass_hinge":
    return multiclass_hinge(logits, one_hot_labels, margin=FLAGS.margin)
  elif FLAGS.loss_fn == "softmax_crossentropy":
    return optax.softmax_cross_entropy(logits * FLAGS.temperature, one_hot_labels)
  else:
    raise ValueError("Unknown loss function.")


class LipschitzCNN(nn.Module):
  """A simple Lipschitz CNN model."""
  num_classes: int
  hidden_widths: Sequence[int] = (16, 16, 32)

  @nn.compact
  def __call__(self, x, train):
    rko_conv = partial(RKOConv, kernel_size=(3, 3), use_bias=FLAGS.use_bias)

    for width in self.hidden_widths[:-1]:
      x = rko_conv(features=width)(x, train=train)
      x = groupsort2(x)
      x = l2norm_pool(x, window_shape=(2, 2), strides=(2, 2))

    if FLAGS.use_global_pool:
      x = rko_conv(features=self.hidden_widths[-1])(x, train=train)
      x = groupsort2(x)
      x = global_l2norm_pool(x)
    else:
      x = jnp.reshape(x, (x.shape[0], -1))
      x = StiefelDense(features=self.hidden_widths[-1], use_bias=FLAGS.use_bias)(x, train=train)
      x = fullsort(x)

    x = Normalized2ToInftyDense(features=self.num_classes, use_bias=FLAGS.use_bias)(x, train=train)

    return x


class LipschitzTrainState(train_state.TrainState):
  """Train state with Lipschitz constraint."""
  lip_state: Any

def create_train_state(rng, num_classes):
  """Creates initial `TrainState`."""
  model = LipschitzCNN(num_classes=num_classes)
  keys = dict(zip(['params', 'lip'], jax.random.split(rng, 2)))
  dummy_batch = jnp.zeros([FLAGS.batch_size, 28, 28, 1])
  model_params = model.init(keys, dummy_batch, train=True)
  params, lip_state = model_params['params'], model_params['lip']

  if FLAGS.dpsgd:
    tx = optax.dpsgd(learning_rate=FLAGS.learning_rate,
                     l2_norm_clip=FLAGS.l2_norm_clip,
                     noise_multiplier=FLAGS.noise_multiplier,
                     seed=12345,
                     momentum=0.9,
                     nesterov=True)
  else:
    tx = optax.adam(learning_rate=FLAGS.learning_rate)

  return LipschitzTrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=tx,
    lip_state=lip_state)


@jax.jit
def update_model(state, grads, lip_vars):
  """Updates model parameters."""
  return state.apply_gradients(grads=grads, lip_state=lip_vars)

@jax.jit
def predict_model(train_state, points):
  """Predicts on a batch of points."""
  model_params = {'params': train_state.params, 'lip': train_state.lip_state}
  preds = train_state.apply_fn(model_params, points, train=False)
  return preds

def tf_to_jax(arr):
  return jnp.array(arr)

def predict_ds(train_state, ds):
  """Predicts on a dataset."""
  jnp_preds = [predict_model(train_state, tf_to_jax(batch)) for batch, _ in ds]
  jnp_preds = jnp.concatenate(jnp_preds)
  return jnp_preds


def evaluate_model(train_state, ds):
  """Evaluates model on a dataset."""
  jnp_preds = predict_ds(train_state, ds)
  jnp_labels = jnp.concatenate([tf_to_jax(label) for _, label in ds], axis=0)
  accuracy = jnp.mean(jnp.argmax(jnp_preds, -1) == jnp_labels)
  loss = jnp.mean(loss_on_batch(logits=jnp_preds, labels=jnp_labels))
  return loss, accuracy


@jax.jit
def apply_model(state, batch):
  """Computes gradients, loss and accuracy for a single batch."""

  def loss_fn(params, example):
    image, label = example['image'], example['label']
    all_params = {'params': params, 'lip': state.lip_state}
    logits, variables = state.apply_fn(all_params, image, train=True, mutable='lip')
    loss = jnp.mean(loss_on_batch(logits=logits, labels=label))
    return loss, (variables['lip'], logits)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

  if FLAGS.dpsgd:
    # Insert dummy dimension in axis 1 to use jax.vmap over the batch
    batch = jax.tree_util.tree_map(lambda x: x[:, None], batch)
    # Use jax.vmap across the batch to extract per-example gradients
    grad_fn = jax.vmap(grad_fn, in_axes=(None, 0), out_axes=0)

  aux, grads = grad_fn(state.params, batch)
  (losses, (lip_vars, logits)) = aux

  if FLAGS.dpsgd:
    # Remove dummy dimension on logits and loss, and Lipschitz variables
    loss = jnp.mean(losses)
    logits = jnp.squeeze(logits, axis=1)
    batch = jax.tree_util.tree_map(lambda x: x.squeeze(axis=1), batch)
    lip_vars = jax.tree_util.tree_map(lambda x: x[0,...], lip_vars)

  accuracy = jnp.mean(jnp.argmax(logits, -1) == batch['label'])
  return grads, (lip_vars, accuracy), loss

@jax.jit
def update_model(state, grads, lip_vars):
  return state.apply_gradients(grads=grads, lip_state=lip_vars)


def train_epoch(state, train_ds,):
  """Train for a single epoch."""
  epoch_loss = []
  epoch_accuracy = []

  for batch_images, batch_labels in (pb := tqdm(train_ds)):
    batch_images = tf_to_jax(batch_images)
    batch_labels = tf_to_jax(batch_labels)
    batch = {'image': batch_images, 'label': batch_labels}
    grads, aux, loss = apply_model(state, batch)
    lip_vars, accuracy = aux
    state = update_model(state, grads, lip_vars=lip_vars)
    epoch_loss.append(loss)
    epoch_accuracy.append(accuracy)
    pb.set_postfix({'loss': float(loss), 'accuracy': float(accuracy*100)})
  
  train_loss = jnp.array(epoch_loss).mean()
  train_accuracy = jnp.array(epoch_accuracy).mean()
  return state, train_loss, train_accuracy


def get_datasets():
  """Load MNIST train and test datasets into memory."""
  train_ds = tfds.load('mnist', split='train', shuffle_files=True)
  test_ds = tfds.load('mnist', split='test', shuffle_files=True)

  def normalize_img(img_label):
    """Normalizes images: `uint8` -> `float32`."""
    image, label = img_label['image'], img_label['label']
    return tf.cast(image, dtype=tf.dtypes.float32) / 255., label
  
  train_ds = train_ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
  train_ds = train_ds.shuffle(1024).batch(FLAGS.batch_size)
  test_ds = test_ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
  test_ds = test_ds.batch(FLAGS.batch_size_test)

  return train_ds, test_ds


def compute_TAN():
  """Compute Total Amount of Noise of the model.
  
  Inspire from paper:
    Sander, T., Stock, P. and Sablayrolles, A., 2022.
    TAN without a burn: Scaling Laws of DP-SGD.
    arXiv preprint arXiv:2210.03403.
  """
  q = FLAGS.batch_size / NUM_EXAMPLES
  sigma = FLAGS.noise_multiplier
  num_steps_per_epoch = NUM_EXAMPLES // FLAGS.batch_size
  S = FLAGS.num_epochs / num_steps_per_epoch
  eta = q / sigma * (S / 2)**0.5
  tan = 1 / eta
  print(f'Total Amount of Noise: {tan:.3f}')


def main(_) :
  """Execute model training and evaluation loop."""
  # Disable GPU for tensorflow.
  tf.config.experimental.set_visible_devices([], 'GPU')

  compute_TAN()

  train_ds, test_ds = get_datasets()
  rng = jax.random.PRNGKey(0)

  rng, init_rng = jax.random.split(rng)
  state = create_train_state(init_rng, num_classes=10)

  for epoch in range(1, FLAGS.num_epochs + 1):
    state, train_loss, train_accuracy = train_epoch(state, train_ds)
    test_loss, test_accuracy = evaluate_model(state, test_ds)

     # Determine privacy loss so far
    if FLAGS.dpsgd:
      steps = epoch * NUM_EXAMPLES // FLAGS.batch_size
      eps = compute_epsilon(steps, FLAGS.delta)
      dp_sgd_msg = f'DP: (delta={FLAGS.delta:.0e}, epsilon={eps:.2f})'
    else:
      dp_sgd_msg = 'DP: (delta=infty, epsilon=infty)'

    logging.info(
        f'epoch:{epoch:3d} | '
        f'train_loss: {train_loss:.5f} '
        f'train_accuracy: {train_accuracy*100:.2f}% | '
        f'test_loss: {test_loss:.5f} '
        f'test_accuracy: {test_accuracy*100:.2f}% | '
        + dp_sgd_msg)


if __name__ == '__main__':
  app.run(main)
