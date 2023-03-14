# This file is part of CNQR which is released under the Apache License 2.0.
# See file LICENSE in the root directory or https://opensource.org/licenses/Apache-2.0 for full license details.

"""Losses.

Design notes:
- Loss functions expect labels and predictions as arguments.
- The batch_size dimension is assumed to be the first dimension.

TODO: decide once for all if the loss function should perform the reduction or not.
"""

import jax
import jax.nn as nn
import jax.numpy as jnp
import jaxopt


def balanced_KR(preds, labels, has_aux=False, eps=1e-2):
  """Balanced Kantorovich-Rubinstein (KR) loss.

  The loss is balanced in the sense that it is invariant to the
  number of positive and negative examples in the batch. This reduces
  the variance of the Monte Carlo estimator of the gradient.
  However when the class imbalance is large, there is a risk that 
  a class is not represented in the batch.

  Hence this loss is not recommended for highly imbalanced datasets.
  This loss is meant to counterbalance stochasticiy across batches
  and mild class imbalance.  
  
  Args:
    preds: array of shape (batch_size,).
    labels: array of shape (batch_size,). Each entry must be either 1 (resp. -1) for P (resp. Q).
    has_aux: whether to return auxiliary values.
    eps: small constant to avoid division by zero.

  Returns:
    loss: float.
    aux: pair of arrays of shape (batch_size,) if has_aux is True.
      Each array contains the normaized contribution of each example to the loss.
  """
  is_P = (labels == 1).astype(jnp.float32)  # size (B,)
  is_Q = (labels == -1).astype(jnp.float32)
  pred_P = preds * is_P / (is_P.sum() + eps)  # averaging over the batch
  pred_Q = preds * is_Q / (is_Q.sum() + eps)
  loss = -(jnp.sum(pred_P) - jnp.sum(pred_Q))  # maximize E_Pf - E_Qf
  if has_aux:
    return loss, (pred_P, pred_Q)
  return loss


def multiclass_hinge(preds, labels, margin):
  """Multiclass hinge loss.

  Args:
    preds: array of shape (batch_size, num_classes).
    labels: array of shape (batch_size, num_classes), one_hot encoding of labels.
    margin: float, margin for the hinge loss.

  Returns:
    loss: tensor of shape (batch_size,). Each entry is the hinge loss for one example.
  """
  pos = jnp.sum(labels * preds, axis=-1)  # size (B,)

  # remove the positive score from the predictions.
  preds_masked_pos = jnp.where(labels > 0., -jnp.inf, preds)

  # retrieve highest score among the negative classes.
  neg = jnp.max(preds_masked_pos, axis=-1)  # size (B,)

  # compute hinge loss.
  elementwise = nn.relu(margin - (pos - neg))

  return elementwise
