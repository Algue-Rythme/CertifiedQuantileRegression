# This file is part of CNQR which is released under the Apache License 2.0.
# See file LICENSE in the root directory or https://opensource.org/licenses/Apache-2.0 for full license details.

"""Losses.

Set of useful losses.
"""

import jax.numpy as jnp


def balanced_KR_W1(labels, preds, has_aux=False, eps=1e-2):
  """Balanced Kantorovich-Rubinstein (KR) loss for Wasserstein-1 distance.

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
      Each array contains the normaized contribution of each example to the loss in terms P and Q.
  """
  is_P = (labels == 1).astype(jnp.float32)  # size (B,)
  is_Q = (labels == -1).astype(jnp.float32)
  fP = preds * is_P / (is_P.sum() + eps)  # averaging over the batch
  fQ = preds * is_Q / (is_Q.sum() + eps)
  loss = -(jnp.sum(fP) - jnp.sum(fQ))  # maximize E_Pf - E_Qf
  if has_aux:
    return loss, (fP, fQ)
  return loss


def symmetric_KR_W2(fP, f_star_Q):
  """Symmetric Kantorovich-Rubinstein (KR) loss for Wasserstein-2 distance.

  The loss is symmetric in the sense that fP and f_star_Q are expected to be of the same size.
  fP is the evaluation of convex potential f on P.
  f_star_Q is the evaluation of the convex conjugate function f* on Q.
  In practice, f_star_Q is only an approximation of f* where the solution x^* of the convex conjugate problem
  is given by the convex potential g.
  
  Args:
    fP: array of shape (batch_size,).
    f_star_Q: array of shape (batch_size,).

  Returns:
    loss: float.
  """
  loss = -jnp.mean(fP) - jnp.mean(f_star_Q)
  return loss
