# This file is part of CNQR which is released under the Apache License 2.0.
# See file LICENSE in the root directory or https://opensource.org/licenses/Apache-2.0 for full license details.

"""Implement of Lipschitz neural network layer in Jax.

Freely inspired by Deel.Lip package: https://github.com/deel-ai/deel-lip distributed under MIT licence.
"""

from functools import partial
from re import T
from typing import Any, Callable, Iterable, List
from typing import Optional, Sequence, Tuple, Union

import jax
from jax import lax
import jax.numpy as jnp
import jax.nn.initializers as initializers
from jax.tree_util import tree_map

from jaxopt.tree_util import tree_zeros_like

import flax
import flax.linen as nn

from cnqr.loop import while_loop


# Useful values found after tests in float32 arithmetic on GPU.
EPSILON_NORMALIZATION = 1e-10
TOL_SPECTRAL_DEFAULT = 1e-5
TOL_BJORCK_DEFAULT = 1e-7
MAXITER_SPECTRAL_DEFAULT = 100
MAXITER_BJORCK_DEFAULT = 100

# According to benchmarks, implicit differentiation is faster 
# than unrolling when the size of the matrix exceeds 1000.
IMPLICIT_OVER_UNROLL = 1000


def l2_normalize(vec, epsilon=EPSILON_NORMALIZATION):
  norm = jnp.linalg.norm(vec, ord=2) + epsilon
  return vec / norm


def power_iteration(W,
                    u_init=None,
                    tol=TOL_SPECTRAL_DEFAULT,
                    maxiter=MAXITER_SPECTRAL_DEFAULT,
                    unroll=True, jit=True):
  """Power iteration for finding the largest eigenvalue and eigenvector.

  Note: it returns a lower bound sigma of the true spectral norm.
    It will only be accurate if `u_init` is not in the null space of the matrix.

  See http://blog.mrtz.org/2013/12/04/power-method.html
  for fun facts about power iteration.
  Don't forget to celebrate its 100th birthday in 2029.
  
  Args:
    W: array of shape (A, B)
    u_init: initial vector of shape (B). Default the mean of the rows of `W`.
      Must not belong to the null space of `W`.
      The closer to the eigenvector associated to the leading eigenvalue it is, the faster the convergence will be.
    maxiter: maximum number of iterations
    unroll: whether to unroll the loop (default: False)
    jit: whether to use jit (default: True)
  Returns:
    right eigenvector, left eigenvector
  """

  if u_init is None:
    u_init = jnp.mean(W, axis=0)

  def cond_fun(t):
    _, error = t
    return error >= tol
  
  def body_fun(t):
    u, _ = t
    v = l2_normalize(jnp.matmul(W, u))
    next_u = l2_normalize(jnp.matmul(v, W))
    error = jnp.linalg.norm(u - next_u)
    # We use stop_gradient to ensure that error is independent of u.
    # Otherwise the JVP rule of scan yields NaNs.  
    # This is a weird bug that must be investigated.
    error = jax.lax.stop_gradient(error)
    return next_u, error

  u_init = l2_normalize(u_init)
  init_state = (u_init, jnp.inf)
  
  u, error = while_loop(cond_fun=cond_fun,
                        body_fun=body_fun,
                        init_val=init_state,
                        maxiter=maxiter,
                        unroll=unroll, jit=jit)

  return u


def spectral_normalization(W,
                           u_init=None,
                           tol=TOL_SPECTRAL_DEFAULT,
                           maxiter=MAXITER_SPECTRAL_DEFAULT,
                           unroll=True, jit=True):
  """Spectral normalization of matrix.

  It computes the leading eigenvector and the leading eigenvalue of :math:`W^T W` to
  obtain the leading singular value of :math:`W`.

  Based on simplified discussion: https://math.stackexchange.com/questions/1255709/finding-the-largest-singular-value-easily
  
  Args:
    W: array of shape (A, B).
    u_init: initial vector of shape (B). See `power_iteration` for details.
    tol: relative tolerance for stopping criterion.
    maxiter: maximum number of iterations.
    unroll: whether to unroll the loop (default: False).
    jit: whether to use jit (default: True).
  
  Returns:
    normalized `W`, unitary leading eigenvector `u` of `W^TW` (eigenvector associated to leading singular value), maximum singular value `sigma`.

    `sigma` is slightly under-estimated, so the normalized `W` returned can have
    a spectral radius slightly greater than one.
  """
  u = power_iteration(W, u_init, tol=tol, maxiter=maxiter, unroll=unroll, jit=jit)
  v = l2_normalize(jnp.matmul(W, u))
  sigma = jnp.vdot(v, jnp.matmul(W, u))  # square root of Rayleigh quotient.
  unit_spectral_norm_matrix = W / (sigma + TOL_SPECTRAL_DEFAULT)
  return unit_spectral_norm_matrix, u, sigma


def bjorck_projection(W,
                      tol=TOL_BJORCK_DEFAULT,
                      maxiter=MAXITER_BJORCK_DEFAULT,
                      beta=0.5, order=1,
                      unroll=True, jit=True):
  """Bjorck algorithm [1] for finding the nearest orthogonal matrix.

  The closer the matrix is to being orthogonal, the faster the convergence will be.

  Original Bjorck's algorithm corresponds to :math:`\beta=0.5\ :
    For order = :math:`\infty`, convergence is only guaranteed if the spectral norm of the matrix is not greater than 1.
    For order 1, convergence is only guaranteed if the spectral norm of `W` is smaller than :math:`\sqrt(3)`. Convergence is significantly faster if the spectral norm is smaller than 1.
  
  Other values of beta correspond to Weight Decay obtained by orthogonal regularization
  with :math:`\beta*\|K^TK-I\|_2^2` like in Parseval's network [2].

  When the matrix is not square, it produces a set of orthogonal vectors of unit norm, i.e a projection onto the Stiefel manifold.

  [1] Björck, Å. and Bowie, C., 1971.
  An iterative algorithm for computing the best estimate of an orthogonal matrix.
  SIAM Journal on Numerical Analysis, 8(2), pp.358-364.
  https://epubs.siam.org/doi/epdf/10.1137/0708036

  [2] Cisse, M., Bojanowski, P., Grave, E., Dauphin, Y. and Usunier, N., 2017, July.
  Parseval networks: Improving robustness to adversarial examples.
  In International Conference on Machine Learning (pp. 854-863). PMLR.
  
  Args:
    W: array of shape (A, B).
    tol: relative tolerance for stopping criterion.
    maxiter: maximum number of iterations.
    beta: beta parameter for the algorithm (default: 0.5)
    order: order of Bjorck's algorithm. (default: 1). Higher values currently not unsupported.
    unroll: whether to unroll the loop (default: False).
    jit: whether to use jit (default: True).
      
  Returns:
      nearest orthogonal matrix of shape (A, B). When A <= B (resp. B >= A) the rows (resp. columns) are orthogonal.
  """
  assert order == 1, "Higher order Bjorck's algorithm is not supported yet."

  wide_matrix = W.shape[0] < W.shape[1]
  if wide_matrix:
    W = W.T
  # Now we can assume B < A in every case.

  def cond_fun(state):
    _, approx_I = state
    I = jnp.eye(len(approx_I))  # shape (B, B)
    res = approx_I - I
    error = jnp.sum(res**2) ** 0.5
    return error >= tol * len(approx_I)
  
  def body_fun(state):
    theta, _ = state
    a = (1 + beta) * theta
    approx_I = theta.T @ theta  # cost: AB^2
    b = beta * theta @ approx_I  # cost: AB^2
    theta = a - b
    return theta, approx_I

  init_state = W, W.T @ W

  state = while_loop(cond_fun=cond_fun,
                     body_fun=body_fun,
                     init_val=init_state,
                     maxiter=maxiter,
                     unroll=unroll, jit=jit)
  
  orthogonal_matrix, _ = state

  if wide_matrix:
    orthogonal_matrix = orthogonal_matrix.T
  
  return orthogonal_matrix


def projection_stiefel_manifold(matrix,
                                u_init=None,
                                tol_spectral=TOL_SPECTRAL_DEFAULT,
                                tol_bjorck=TOL_BJORCK_DEFAULT,
                                maxiter_spectral=MAXITER_SPECTRAL_DEFAULT,
                                maxiter_bjorck=MAXITER_BJORCK_DEFAULT,
                                unroll=True, jit=True):
    """Orthogonalize matrix using spectral normalization and Bjorck projection.

    Warning:
      The Stiefel manifold is not convex. Hence some algorithms that relies on convex projections might fail unexpectedly with orthogonal projections.
      It is not connected either: it contains two connected components of determinant +1 and -1 respectively for square matrices.  

    If back-propagation through iterations is too slow, consider using implicit differentiation instead with implicit_differentiation_stiefel_proj wrapper.

    Args:
      matrix: array of shape (A, B).
      u_init: guess for leading eigenvector. See documentation of `power_iteration`.
      tol_spectral: relative tolerance for stopping criterion.
      tol_bjorck: relative tolerance for stopping criterion.
      maxiter_spectral: maximum number of iterations.
      maxiter_bjorck: maximum number of iterations.
      unroll: whether to unroll the loop (default: False).
      jit: whether to use jit (default: True).

    Returns:
      nearest orthogonal matrix of shape (A, B), unitary leading eigenvector `u` (eigenvector associated to leading eigenvalue), maximum singular value `sigma`.
    """
    normalized_matrix, u, sigma = spectral_normalization(matrix, u_init,
                                                         tol=tol_spectral,
                                                         maxiter=maxiter_spectral,
                                                         unroll=unroll, jit=jit)
    orthogonal_matrix = bjorck_projection(normalized_matrix,
                                          tol=tol_bjorck,
                                          maxiter=maxiter_bjorck,
                                          unroll=unroll, jit=jit)
    return orthogonal_matrix, u, sigma


def implicit_differentiation_stiefel_proj(proj_fun, solver='eigen', has_aux=False):
  """Decorator that wraps any projection onto the Stiefel manifold with implicit differentiation.

  Jaxopt default pipeline is too slow and significant speed-up is achieved by this implementation that leverages specificities
  of Stiefel manifold and the particular form of KKT system.

  Warning:
    Implicit differentiation solves a system ill-posed when there exists a pair of eigenvalues (i,j) such that :math:`\sigma_i+\sigma_j=0`.
    In particular this happens when W is singular, or when the spectrum of W is symmetric around origin.
    These failure cases are of measure zero in the set of matrices. However they happen frequently in practice on problems that exhibit some symmetries.
    For example, this happens if W is exactly half way between the two connected components of orthogonal matrices manifold. In this case, the projection is not well defined.  
    The user must check that its problem is well defined in this case; otherwise the gradient returned is guaranteed to be not `nan` but may be inaccurate.

  Args:
    proj_fun: function that takes a matrix W and returns an orthogonal matrix (if has_aux=False)
              or an orthogonal matrix and auxiliary information (if has_aux=True).
    solver: solver to use for the KKT system (default: 'eigen').
      'eigen' uses eigen decomposition.
      'cg' uses conjugate gradient.
    has_aux: whether the projection function returns auxiliary information (default: False). Auxiliary information is ignored in backward pass.

  Returns:
    A function that takes a matrix W and returns an orthogonal matrix (if has_aux=False), or an orthogonal matrix and auxiliary information (if has_aux=True).
  """

  @jax.custom_vjp
  def implicit_proj_fun(W):
    t = proj_fun(W)
    return t

  def implicit_proj_fun_fwd(W):
    t = implicit_proj_fun(W)
    if has_aux:
      theta = t[0]  # extract relevant information.
    else:
      theta = t
    return t, (theta, W)

  def implicit_proj_fun_bwd(res, t_grad):
    theta, W = res

    if has_aux:
      theta_grad = t_grad[0]
      # auxilary information is not used in the backward pass currently.

    # Assume W = U Sigma V^T is the SVD of W.
    # Then theta = U V^T is the solution to the orthogonal procruste problem.

    B = W.T @ theta  # B = V Sigma V^T.
    OTu = theta.T @ theta_grad
    C = OTu - OTu.T  # C is skew symmetric.

    if solver == 'cg':
      
      def anticommutator(X):
        return B@X + X@B
      
      A, _ = jax.scipy.sparse.linalg.cg(anticommutator, C)

    elif solver == 'eigen':
      # Based on: https://math.stackexchange.com/questions/528831/solutions-to-the-anticommutator-matrix-equation
      Sigma, V = jnp.linalg.eigh(B)
      VT = V.T

      lbdas = Sigma[jnp.newaxis,:] + Sigma[:,jnp.newaxis]
      # handle singularities.
      epsilon = 1e-9
      lbdas = jnp.where(jnp.abs(lbdas) <= epsilon, epsilon, lbdas)

      C_tilde = VT @ C @ V
      A_tilde = C_tilde / lbdas
      A = V @ A_tilde @ VT
    
    # tangeant vector: A is a skew symmetric matrix.
    x = theta @ A

    return (x,)

  implicit_proj_fun.defvjp(implicit_proj_fun_fwd, implicit_proj_fun_bwd)

  return implicit_proj_fun


class StiefelDense(nn.Module):
  """Dense layer with orthogonality constraint enforced by projection onto Stiefel manifold.

  If the kernel is square then it is orthogonal.
  If there is more rows than than columns then the columns are orthogonal.
  If there is more columns than rows then the rows are orthogonal.

  Orthogonalization is performed using Power Iteration and Bjorck algorithm.

  Attributes:
    features: number of output features.
    train: whether to train the layer (default: True).
    kernel_init: initializer for the weight matrix (default: orthogonal).
    bias_init: initializer for the bias (default: zero).
    u_init: guess for leading eigenvector. See documentation of `power_iteration`.
    implicit_diff: whether to use implicit differentiation (default: "auto").
      Behavior "auto" switches to True if the kernel is square and big enough (default: bigger than 1000).
    tol_spectral: relative tolerance for stopping criterion of spectral normalization.
    tol_bjorck: relative tolerance for stopping criterion of Bjorck algorithm.
    maxiter_spectral: maximum number of iterations for spectral normalization.
    maxiter_bjorck: maximum number of iterations for Bjorck algorithm.
  """
  features: int
  maxiter_spectral: Optional[int] = None
  maxiter_bjorck: Optional[int] = None
  train: Optional[bool] = None
  use_bias: bool = True
  kernel_init: Callable = initializers.orthogonal()
  bias_init: Callable = initializers.zeros
  u_init: Callable = jax.random.normal
  implicit_diff: Union[str, bool] = "auto"
  tol_spectral: float = TOL_SPECTRAL_DEFAULT
  tol_bjorck: float = TOL_BJORCK_DEFAULT
  maxiter_spectral: int = MAXITER_SPECTRAL_DEFAULT
  maxiter_bjorck: int = MAXITER_BJORCK_DEFAULT

  def _stiefel_projection(self, kernel, u):
    """Project onto Stiefel manifold."""

    if isinstance(self.implicit_diff, str) and self.implicit_diff == "auto":
      is_square = kernel.shape[0] == kernel.shape[1]
      large_matrix = kernel.shape[0] > IMPLICIT_OVER_UNROLL
      implicit_diff = is_square and large_matrix  # matrix too big: implicit differentiation mandatory.
    else:
      implicit_diff = self.implicit_diff

    def stiefel_proj(W):
      return projection_stiefel_manifold(
        W, u_init=u,  # closure over u: considered constant.
        tol_spectral=self.tol_spectral,
        tol_bjorck=self.tol_bjorck,
        maxiter_spectral=self.maxiter_spectral,
        maxiter_bjorck=self.maxiter_bjorck,
        unroll=not implicit_diff,
        jit=True
      )

    if implicit_diff:
      stiefel_proj = implicit_differentiation_stiefel_proj(stiefel_proj, solver='eigen', has_aux=True)

    return stiefel_proj(kernel)

  @nn.compact
  def __call__(self, inputs, train=None):
    """Forward pass.

    Args:
      inputs: array of shape (B, f_1) with B the batch_size.
      train: whether to use perform orthogonalization of re-use the cached kernel.

    Returns:
      outputs: array of shape (B, features)
    """

    # init params
    kernel_shape = (inputs.shape[-1], self.features)
    kernel = self.param('kernel', self.kernel_init, kernel_shape)
    if self.use_bias:
      bias = self.param('bias', self.bias_init, (1, self.features))

    # init mutable variables
    orthogonal_kernel = self.variable('lip', 'orthogonal_kernel', jnp.zeros, kernel_shape)
    u = self.variable('lip', 'u',
      (lambda shape: l2_normalize(self.u_init(self.make_rng('lip'), shape))), (kernel_shape[-1],))
    sigma = self.variable('lip', 'sigma', lambda shape: jnp.zeros(shape), ())

    train = nn.merge_param('train', self.train, train)
    if train:
      ortho_ker, _u, _sigma = self._stiefel_projection(kernel, u.value)
      # cache the value for future use.
      orthogonal_kernel.value = ortho_ker
      u.value = _u
      sigma.value = _sigma
    else:
      ortho_ker = orthogonal_kernel.value

    y = jnp.matmul(inputs, ortho_ker)
    if self.use_bias:
      y = y + bias

    return y


##############################################
############## Activations ###################
##############################################


def channelwise_groupsort2(x):
  """GroupSort2 activation function.

  Args:
      x: array of shape (C,). C must be an even number.

  Returns:
      array of shape (C,) with groupsort2 applied.
  """
  assert x.shape[0] % 2 == 0
  a, b = jnp.split(x, 2, axis=0)
  min_ab = jnp.minimum(a, b)
  max_ab = jnp.maximum(a, b)
  return jnp.concatenate([min_ab, max_ab], axis=0)


def groupsort2(x):
  """GroupSort2 activation function.

  Args:
      x: array of shape (B, C). C must be an even number.

  Returns:
      array of shape (B, C) with groupsort2 applied.
  """
  return jax.vmap(channelwise_groupsort2, in_axes=0)(x)


def full_sort(x):
  """FullSort activation function.

  Args:
      x: array of shape (B, C).

  Returns:
      array of shape (B, C) with full sort applied: dimension C is sorted.
  """
  return jnp.sort(x, axis=-1)
