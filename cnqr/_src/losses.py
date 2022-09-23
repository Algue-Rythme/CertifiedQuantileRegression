# This file is part of CNQR which is released under the Apache License 2.0.
# See file LICENSE in the root directory or https://opensource.org/licenses/Apache-2.0 for full license details.

"""Losses.

Set of useful losses.
"""

import jax.numpy as jnp


def balanced_KR(labels, preds, has_aux=False, eps=1e-2):
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
    labels: array of shape (batch_size,). Each entry must be either 1 (resp. -1) for P (resp. Q).
    preds: array of shape (batch_size,).
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
