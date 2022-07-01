# Borrowed from Jaxopt library (distributed under Apache License 2.0)
# See: https://github.com/google/cnqr/blob/main/jaxopt/tree_util.py 

"""Tree utilities."""

from cnqr._src.tree_util import broadcast_pytrees
from cnqr._src.tree_util import tree_map
from cnqr._src.tree_util import tree_reduce
from cnqr._src.tree_util import tree_add
from cnqr._src.tree_util import tree_sub
from cnqr._src.tree_util import tree_mul
from cnqr._src.tree_util import tree_scalar_mul
from cnqr._src.tree_util import tree_add_scalar_mul
from cnqr._src.tree_util import tree_dot
from cnqr._src.tree_util import tree_vdot
from cnqr._src.tree_util import tree_div
from cnqr._src.tree_util import tree_sum
from cnqr._src.tree_util import tree_l2_norm
from cnqr._src.tree_util import tree_where
from cnqr._src.tree_util import tree_zeros_like
from cnqr._src.tree_util import tree_ones_like
from cnqr._src.tree_util import tree_negative
from cnqr._src.tree_util import tree_inf_norm
