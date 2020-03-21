import random
from collections import defaultdict

import dwave_networkx as dnx
import networkx as nx

from .utils import dnx_utils, layout_utils, placement_utils


def neighborhood(placement, second=False, **kwargs):
    """
    Given a placement (a map, phi, from vertices of S to the subsets of vertices of T), form the chain N_T(phi(u)) 
    (closed neighborhood of v in T) for each u in S.

    Parameters
    ----------
    placement : placement.Placement
        A mapping from vertices of S (keys) to subsets of vertices of T (values).
    second : bool (default False)
        If True, gets the closed 2nd neighborhood of each vertex. If False, get the closed 1st neighborhood of each
        vertex.

    Returns
    -------
    placement : placement.Placement
        A mapping from vertices of S (keys) to subsets of vertices of T (values).
    """
    placement.chains = {u: _closed_neighbors(placement.T, V, second=second)
                        for u, V in placement.items()}
    return placement


def _closed_neighbors(G, U, second):
    """
    Returns the closed neighborhood of u in G.

    Parameters
    ----------
    G : NetworkX graph
        The graph you are considering.
    U : list or frozenset or set
        A collection of nodes you are computing the closed neighborhood of.
    second : bool
        If True, compute the closed 2nd neighborhood.

    Returns
    -------
    neighbors: set
        A set of vertices of G.
    """
    closed_neighbors = nx.node_boundary(G, U) | set(U)

    if second:
        return nx.node_boundary(G, closed_neighbors) | closed_neighbors
    return closed_neighbors
