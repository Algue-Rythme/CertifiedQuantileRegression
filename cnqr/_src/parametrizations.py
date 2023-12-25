# This file is part of CNQR which is released under the Apache License 2.0.
# See file LICENSE in the root directory or https://opensource.org/licenses/Apache-2.0 for full license details.

from functools import partial
from re import T
from typing import Any, Callable, Iterable, List
from typing import Optional, Sequence, Tuple, Union

import jax
from jax import lax
import jax.numpy as jnp
import jax.nn.initializers as initializers
from jax.tree_util import tree_map

import jaxopt.projection
from jaxopt.tree_util import tree_zeros_like, tree_add, tree_sub

import flax
import flax.linen as nn

from cnqr.loop import while_loop


EPSILON_NORMALIZATION = 1e-10

BJORCK_PRECISION = "medium"
if BJORCK_PRECISION == "low":
  # Adapted from deel-lip 1.4.0
  TOL_SPECTRAL_DEFAULT = 1e-3
  # in deel-lip it's ATOL=1e-3, while here it's RTOL=1e-5 wrt matrix length.
  # This covers the case len(M) ~= 256 frequently encountered in modern architectures.
  TOL_BJORCK_DEFAULT = 1e-3 / 256   
  MAXITER_SPECTRAL_DEFAULT = 10
  MAXITER_BJORCK_DEFAULT = 15
elif BJORCK_PRECISION == "medium":
  # Useful values found after tests in float32 arithmetic on GPU.
  TOL_SPECTRAL_DEFAULT = 1e-4
  TOL_BJORCK_DEFAULT = 1e-5
  MAXITER_SPECTRAL_DEFAULT = 30
  MAXITER_BJORCK_DEFAULT = 30
elif BJORCK_PRECISION == "high":
  # Useful values found after tests in float32 arithmetic on GPU.
  # Required to orthogonaize badly conditioned matrices.
  TOL_SPECTRAL_DEFAULT = 1e-5
  TOL_BJORCK_DEFAULT = 1e-7
  MAXITER_SPECTRAL_DEFAULT = 100
  MAXITER_BJORCK_DEFAULT = 100
else:
  raise ValueError(f"Unknown BJORCK_PRECISION={BJORCK_PRECISION}")

# According to benchmarks, implicit differentiation is faster 
# than unrolling when the size of the matrix exceeds 1000.
IMPLICIT_OVER_UNROLL = 1000


class CachedParametrization(nn.Module):
  """Base class for cached parametrizations.
  
  This kind of parametrization is designed to be used in forward pass
  of neural networks. The parametrization is cached during the forward pass
  when training=True and re-used during the forward pass when training=False.

  The cached results are stored in the state of the module using Flax variables.
  They are not part of the parameters of the module and are not updated by the optimizer.

  The parametrization is stateful so it inherits from nn.Module, it may requires a key for RNGs.
  Note that "lights" re-parametrizations that do not require memoization should be implemented
  as a simple functions.

  Warning: when training=False, the cached results are used and not recomputed. In particular backpropagation
  to unconstrained parameters is not possible. Moreover, the cached parameters "lag" behind the unconstrained ones
  so at the end of training there is a gradient step difference between Pi(W_{t+1}) and V_t,
  where Pi is the parametrization and V_t = Pi(W_t) is the constrained value.

  The following arguments are common to all parametrizations:

  Args:
    train: whether to compute the differentiable parametrization or use the cached one.
    auto_diff: str, either 'auto', 'unroll', 'implicit' or 'identity'.
      If 'unroll', the parametrization is unrolled by backpropagation during the forward pass.
      If 'implicit', the implicit differentiation is used.
      If 'identity', the parametrization is treated as an identity.
      If 'auto', the differentiation policy is automatically selected based on the results of a benchmark.
      Not all parametrizations support all differentiation policies. Check the documentation of the parametrization.
    groupname: str, name of the group of parameters to which the parametrization is applied.
  """
  pass


def straight_through(fun):
  """Straight-through estimator for identity parametrization."""

  def _straight_through(tree):
    # Create an exactly-zero expression with Sterbenz lemma that has
    # an exactly-one gradient.
    zero = tree_sub(tree, jax.lax.stop_gradient(tree))
    return tree_add(zero, jax.lax.stop_gradient(fun(tree)))
  
  return _straight_through


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
    error = jnp.linalg.norm(u - next_u)  # absolute error.
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
  Note that the left or right multiplication convention is irrelevant here since ||W||_2 = ||W^T||_2.

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
  r"""Bjorck algorithm [1] for finding the nearest orthogonal matrix.

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
    return error >= tol * len(approx_I)  # relative error.
  
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


def projection_spectral_bjorck(matrix,
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
  r"""Decorator that wraps any projection onto the Stiefel manifold with implicit differentiation.

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


class BjorckParametrization(CachedParametrization):
  """Base class for Bjorck parametrizations.
  
  Args:
    train: whether to train the parametrization (default: True).
    auto_diff: support 'auto' (default), 'unroll', 'implicit' and 'identity' modes.
      See documentation of CachedParametrization for more details.
    u_init: guess for leading eigenvector. See documentation of `power_iteration`.
    tol_spectral: relative tolerance for stopping criterion of spectral normalization.
    tol_bjorck: relative tolerance for stopping criterion of Bjorck algorithm.
    maxiter_spectral: maximum number of iterations for spectral normalization.
    maxiter_bjorck: maximum number of iterations for Bjorck algorithm.
  """
  train: Optional[bool] = None
  auto_diff: str = 'auto'
  groupname: str = 'lip'
  u_init: Callable = jax.random.normal
  tol_spectral: float = TOL_SPECTRAL_DEFAULT
  tol_bjorck: float = TOL_BJORCK_DEFAULT
  maxiter_spectral: int = MAXITER_SPECTRAL_DEFAULT
  maxiter_bjorck: int = MAXITER_BJORCK_DEFAULT

  def _stiefel_projection(self, kernel, u):
    """Project onto Stiefel manifold."""

    auto_diff = self.auto_diff
    if auto_diff == "auto":
      is_square = kernel.shape[0] == kernel.shape[1]
      large_matrix = kernel.shape[0] > IMPLICIT_OVER_UNROLL
      if is_square and large_matrix:  # matrix too big: implicit differentiation mandatory.
        auto_diff = "implicit"
      else:
        auto_diff = "unroll"

    unroll = auto_diff == "unroll"  # unravel the loop.
    def stiefel_proj(W):
      return projection_spectral_bjorck(
        W, u_init=u,  # closure over u: considered constant.
        tol_spectral=self.tol_spectral,
        tol_bjorck=self.tol_bjorck,
        maxiter_spectral=self.maxiter_spectral,
        maxiter_bjorck=self.maxiter_bjorck,
        unroll=unroll,
        jit=True
      )

    if auto_diff == "implicit":
      stiefel_proj = implicit_differentiation_stiefel_proj(stiefel_proj, solver='eigen', has_aux=True)
    elif auto_diff == "identity":
      stiefel_proj = straight_through(stiefel_proj)
    else:
      pass # no need to do anything.

    return stiefel_proj(kernel)

  @nn.compact
  def __call__(self, kernel, train=None):
    """Forward pass.

    Args:
      kernel: array of shape (f_1, f_2) with f_1 the input dimension and f_2 the output dimension.
      train: whether to use perform orthogonalization of re-use the cached kernel.

    Returns:
      outputs: array of shape (B, features)
    """

    # init params
    kernel_shape = kernel.shape

    # init mutable variables
    orthogonal_kernel = self.variable(self.groupname, 'orthogonal_kernel', jnp.zeros, kernel_shape)
    u = self.variable(self.groupname, 'u',
      (lambda shape: l2_normalize(self.u_init(self.make_rng(self.groupname), shape))), (kernel_shape[-1],))
    sigma = self.variable(self.groupname, 'sigma', lambda shape: jnp.zeros(shape), ())

    train = nn.merge_param('train', self.train, train)
    if train:
      ortho_ker, _u, _sigma = self._stiefel_projection(kernel, u.value)
      # cache the value for future use.
      orthogonal_kernel.value = ortho_ker
      u.value = _u
      sigma.value = _sigma
    else:
      ortho_ker = orthogonal_kernel.value

    return ortho_ker


def projection_2_infty_ball(mat, epsilon=1e-9):
  r"""Project matrix on unit ball with 2->inf norm using left-multiplication convention (x |-> xM).

  See [1] for details.

  The 2->inf norm is maximum l2 norm of the columns (Proposition 6.1 p20 of [1]):
    ||A||_{2->inf} = max_{j} ||A_{\cdot,j}||_2

  Per remark 6.2 p20 of [1] for any vector x:
    ||xA||_{inf} <= ||x||_2 ||A||_{2->inf}
  However this property is not true in general when x is a matrix.

  A slightly different result actually holds (Proposition 6.5 p21 of [1]):
    ||BA||_{2->inf} <= ||B||_2        ||A||_{2->inf} 
    ||AC||_{2->inf} <= ||A||_{2->inf} ||B||_{inf}

  Note that the following inequality holds (Proposition 6.3 p20 of [1]):
    ||A||_{2->inf}  <= ||A||_2 <= min(sqrt(N) * ||A||_{2->inf}, sqrt(M) * ||A^T||_{2->inf})

  [1] Cape, J., Tang, M. and Priebe, C.E., 2019.
      The two-to-infinity norm and singular subspace geometry with applications to high-dimensional statistics.
      The Annals of Statistics, 47(5), pp.2405-2439.
      https://arxiv.org/pdf/1705.10735.pdf
  
  Args:
    mat: matrix to normalize of shape (M, N) where M denotes input dimension and N output dimension.
    epsilon: small value to avoid division by zero.
    
  Returns:
    normalized matrix of same shape as mat.
  """
  vec_norms = jnp.sqrt(jnp.sum(mat**2, axis=0, keepdims=True))
  vec_norms = jnp.clip(vec_norms, a_min=epsilon, a_max=None)
  mat = mat / vec_norms  # unit norm columns.
  vec_norms = jnp.clip(vec_norms, a_min=None, a_max=1.)
  return mat * vec_norms  # rescale columns to original norm when their norm is smaller than 1.


def projection_vector_l1_ball(vec):
  return jaxopt.projection.projection_l1_ball(vec, radius=1.)

def projection_matrix_l1_ball(mat):
  r"""Project matrix on unit ball with l-inf norm using left-multiplication convention (x |-> xM).

  The infty norm of a matrix is the maximum absolute column sum of the matrix:

    ||A||_{inf} = max_{j} ||A_{\cdot,j}||_1

  Useful inequalities (from wikipedia):

    1/sqrt(M) ||A||_{inf} <= ||A||_2 <= sqrt(N)||A||_{inf}
  
  Args:
    mat: matrix to normalize of shape (M, N) where M denotes input dimension and N output dimension.
    
  Returns:
    normalized matrix of same shape as mat.
  """
  return jax.vmap(projection_vector_l1_ball)(mat, in_axes=(1,), out_axes=1)


class NormalizationParametrization(CachedParametrization):
  """Normalized matrix weights for well chosen norm.
  
  Attributes:
    normalize_fun: function to normalize the matrix weights.
    train: whether to use perform normalization or re-use the cached kernel.
    auto_diff: support 'auto' (default), 'unroll', 'implicit' and 'identity' modes.
      See documentation of CachedParametrization for more details.
  """
  normalize_fun: Callable
  train: Optional[bool] = None
  groupname: str = 'lip'
  auto_diff: str = 'auto'

  @nn.compact
  def __call__(self, kernel, train=None):
    """Forward pass.

    Args:
      kernel: array of shape (f_1, f_2) with f_1 the input dimension and f_2 the output dimension.
      train: whether to use perform orthogonalization of re-use the cached kernel.

    Returns:
      outputs: array of shape (B, features)
    """
    # init params
    kernel_shape = kernel.shape

    normalized_kernel = self.variable(self.groupname, 'normalized_kernel', jnp.zeros, kernel_shape)

    train = nn.merge_param('train', self.train, train)
    if train:
      normalized_ker = self.normalize_fun(kernel)
      normalized_kernel.value = normalized_ker
    else:
      normalized_ker = normalized_kernel.value

    return normalized_ker


Normalized2ToInftyParametrization = partial(NormalizationParametrization, normalize_fun=projection_2_infty_ball)
NormalizedInftyParametrization = partial(NormalizationParametrization, normalize_fun=projection_matrix_l1_ball)


class PositiveOrthant(CachedParametrization):
  """Tensor with positive weights (> 0).
  
  Attributes:
    beta: positive scalar to scale the weights.
    train: whether to use perform orthogonalization or re-use the cached kernel.
    auto_diff: support 'auto' (default), 'unroll' and 'identity' modes.
      See documentation of CachedParametrization for more details.
  """
  beta = 1.
  train: Optional[bool] = None
  groupname: str = 'convex'
  auto_diff: str = 'auto'

  def inv_act_fun(self, tensor):
    return (1. / self.beta) * jnp.log(jnp.exp(tensor * self.beta) - 1.)

  def act_fun(self, tensor):
    return (1. / self.beta) * nn.softplus(tensor * self.beta)

  @nn.compact
  def __call__(self, tensor, train=None):
    """Forward pass.

    Args:
      tensor: array of arbitrary shape.
      train: whether to use perform orthogonalization or re-use the cached kernel.

    Returns:
      outputs: array of shape (B, features)
    """
    # init params
    tensor_shape = tensor.shape

    # init mutable variables
    positive_tensor = self.variable(self.groupname, 'positive_tensor', jnp.zeros, tensor_shape)

    train = nn.merge_param('train', self.train, train)
    if train:

      auto_diff = self.auto_diff
      if self.auto_diff == 'auto':
        auto_diff = 'unroll'

      act_fun = self.act_fun
      if auto_diff == 'identity':
        act_fun = straight_through(act_fun)

      positive_ker = act_fun(tensor)
      positive_tensor.value = positive_ker
      
    else:
      positive_ker = positive_tensor.value

    return positive_ker
