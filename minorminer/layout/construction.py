from collections import defaultdict

import dwave_networkx as dnx
import minorminer as mm
from minorminer.layout import layout
from minorminer.layout.utils import dnx_utils, layout_utils, placement_utils


def pass_along(placement, **kwargs):
    """
    Given a placement (a map, phi, from vertices of S to vertices (or subsets of vertices) of T), form the chain 
    [phi(u)] (or phi(u)) for each u in S.

    Parameters
    ----------
    placement : dict
        A mapping from vertices of S (keys) to vertices of T (values).

    Returns
    -------
    chains: dict
        A mapping from vertices of S (keys) to chains of T (values).
    """
    # Test if you need to turn singletons into lists or not
    if layout_utils.convert_to_chains(placement):
        chains = {u: [v] for u, v in placement.items()}
    else:
        chains = placement

    return chains


def crosses(placement, S_layout, T, **kwargs):
    """
    Extend chains for vertices of S along rows and columns of qubits of T (T must be a D-Wave hardware graph). 

    If you project the layout of S onto 1-dimensional subspaces, for each vertex u of S a chain is a minimal interval 
    containing all neighbors of u. This amounts to a placement where each chain is a cross shaped chain where the middle 
    of the cross is contained in a unit cell, and the legs of the cross extend horizontally and vertically as far as 
    there are neighbors of u.

    Parameters
    ----------
    placement : dict
        A mapping from vertices of S (keys) to vertices of T (values).
    S_layout : layout.Layout
        A layout for S; i.e. a map from S to R^d.
    T : layout.Layout or dwave-networkx.Graph
        A layout for T; i.e. a map from T to R^d. Or a D-Wave networkx graph to make a layout from.

    Returns
    -------
    chains: dict
        A mapping from vertices of S (keys) to chains of T (values).
    """
    T_layout = placement_utils.parse_T(T, disallow="dict")
    assert isinstance(S_layout, layout.Layout) and isinstance(T_layout, layout.Layout), (
        "Layout class instances must be passed in.")
    assert S_layout.d == 2 and T_layout.d == 2, "This is only implemented for 2-dimensional layouts."

    # Grab the coordinate version of the labels
    if T_layout.G.graph["labels"] == "coordinate":
        chains = dict(placement)
    else:
        n, m, t = dnx_utils.lookup_dnx_dims(T_layout.G)
        C = dnx.chimera_coordinates(m, n, t)
        chains = {v: [C.linear_to_chimera(
            q) for q in Q] for v, Q in placement.items()}

    for v, Q in placement.items():
        hor_v, ver_v = (Q[0], Q[1]) if Q[0][2] == 0 else (Q[1], Q[0])

        min_x, max_x = ver_v[1], ver_v[1]
        min_y, max_y = ver_v[0], ver_v[0]
        for u in S_layout.G[v]:
            QQ = placement[u]
            hor_u, ver_u = (QQ[0], QQ[1]) if QQ[0][2] == 0 else (QQ[1], QQ[0])

            min_x = min(min_x, hor_u[1])
            max_x = max(max_x, hor_u[1])
            min_y = min(min_y, ver_u[0])
            max_y = max(max_y, ver_u[0])

            row_qubits = set()
            for j in range(min_x, max_x+1):
                row_qubits.add((ver_v[0], j, 1, ver_v[3]))

            column_qubits = set()
            for i in range(min_y, max_y+1):
                column_qubits.add((i, hor_v[1], 0, hor_v[3]))

        chains[v] = row_qubits | column_qubits

    # Return the right type of vertices
    if T_layout.G.graph["labels"] == "coordinate":
        return chains
    else:
        return {v: [C.chimera_to_linear(q) for q in Q] for v, Q in chains.items()}


def neighborhood(S, T, placement, second=False, extend=False):
    """
    Given a placement (a map, phi, from vertices of S to vertices of T), form the chain N_T(phi(u)) (closed neighborhood 
    of v in T) for each u in S.

    Parameters
    ----------
    T : NetworkX graph
        The graph you are embedding into (target).
    placement : dict
        A mapping from vertices of S (keys) to vertices of T (values).
    second : bool (default False)
        If True, gets the closed 2nd neighborhood of each vertex. If False, get the closed 1st neighborhood of each
        vertex. 
    extend : bool (default False)
        If True, extend chains to mimic the structure of S in T.

    Returns
    -------
    chains: dict
        A mapping from vertices of S (keys) to chains of T (values).
    """
    chains = {u: closed_neighbors(T, v, second=second)
              for u, v in placement.items()}
    if extend:
        return extend_chains(S, T, chains)

    return chains


def extend_chains(S, T, initial_chains):
    """
    Extend chains in T so that their structure matches that of S. That is, form an overlap embedding of S in T
    where the initial_chains are subsets of the overlap embedding chains. This is done via minorminer.

    Parameters
    ----------
    S : NetworkX graph
        The graph you are embedding (source).
    T : NetworkX graph
        The graph you are embedding into (target).
    initial_chains : dict
        A mapping from vertices of S (keys) to chains of T (values).

    Returns
    -------
    extended_chains: dict
        A mapping from vertices of S (keys) to chains of T (values).
    """
    # Extend the chains to minimal overlapped embedding
    miner = mm.miner(S, T, initial_chains=initial_chains)
    extended_chains = defaultdict(set)
    for u in S:
        # Revert to the initial_chains and compute the embedding where you tear-up u.
        emb = miner.quickpass(
            [u], clear_first=True, overlap_bound=S.number_of_nodes()
        )

        # Add the new chain for u and the singleton one too (in case it got moved in emb)
        extended_chains[u].update(set(emb[u]).union(initial_chains[u]))

        # For each neighbor v of u, grow u to reach v
        for v in S[u]:
            extended_chains[u].update(
                set(emb[v]).difference(initial_chains[v]))

    return extended_chains


def closed_neighbors(G, u, second=False):
    """
    Returns the closed neighborhood of u in G.

    Parameters
    ----------
    G : NetworkX graph
        The graph you are considering.
    u : NetworkX node
        The node you are computing the closed neighborhood of.
    second : bool (default False)
        If True, compute the closed 2nd neighborhood.

    Returns
    -------
    neighbors: set
        A set of vertices of G.
    """
    neighbors = set(v for v in G.neighbors(u))
    if second:
        return set((u,)) | neighbors | set(w for v in neighbors for w in G.neighbors(v))
    return set((u,)) | neighbors
