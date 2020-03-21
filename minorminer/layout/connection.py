import random
from collections import defaultdict

import dwave_networkx as dnx
import minorminer as mm
import networkx as nx

from .utils import dnx_utils, layout_utils, placement_utils


def crosses(placement):
    """
    Extend chains for vertices of S along rows and columns of qubits of T (T must be a D-Wave hardware graph). 

    If you project the layout of S onto 1-dimensional subspaces, for each vertex u of S a chain is a minimal interval 
    containing all neighbors of u. This amounts to a placement where each chain is a cross shaped chain where the middle 
    of the cross is contained in a unit cell, and the legs of the cross extend horizontally and vertically as far as 
    there are neighbors of u.

    Parameters
    ----------
    placement : placement.Placement
        A mapping from vertices of S (keys) to subsets of vertices of T (values).

    Returns
    -------
    placement : placement.Placement
        A mapping from vertices of S (keys) to subsets of vertices of T (values).
    """
    # Currently only implemented for 2d chimera
    if placement.T.graph.get("family") != "chimera" or placement.T_layout.d != 2:
        raise NotImplementedError(
            "This strategy is currently only implemented for Chimera in 2d.")

    # Grab the coordinate version of the labels
    if placement.T.graph["labels"] == "coordinate":
        chains = placement.chains
    else:
        n, m, t = dnx_utils.lookup_dnx_dims(placement.T)
        C = dnx.chimera_coordinates(m, n, t)
        chains = {
            v: {C.linear_to_chimera(q) for q in Q} for v, Q in placement.items()
        }

    for v in placement.S:
        hor_v, ver_v = _horizontal_and_vertical_qubits(chains[v])

        min_x = min(hor_v[1], ver_v[1])
        max_x = max(hor_v[1], ver_v[1])
        min_y = min(hor_v[0], ver_v[0])
        max_y = max(hor_v[0], ver_v[0])
        for u in placement.S[v]:
            hor_u, ver_u = _horizontal_and_vertical_qubits(chains[u])

            min_x = min(min_x, min(hor_u[1], ver_u[1]))
            max_x = max(max_x, max(hor_u[1], ver_u[1]))
            min_y = min(min_y, min(hor_u[0], ver_u[0]))
            max_y = max(max_y, max(hor_u[0], ver_u[0]))

            row_qubits = set()
            for j in range(min_x, max_x+1):
                row_qubits.add((ver_v[0], j, 1, ver_v[3]))

            column_qubits = set()
            for i in range(min_y, max_y+1):
                column_qubits.add((i, hor_v[1], 0, hor_v[3]))

        chains[v] = row_qubits | column_qubits

    # Return the right type of vertices
    placement.chains = dnx_utils.relabel_chains(placement.T, chains)
    return placement


def _horizontal_and_vertical_qubits(chain):
    """
    Given a chain, select one horizontal and one vertical qubit. If one doen't exist, extend the chain to include one. 
    """
    # Split each chain into horizontal and vertical qubits
    horizontal_qubits = [q for q in chain if q[2] == 0]
    vertical_qubits = [q for q in chain if q[2] == 1]

    # FIXME: Making a random choice here might not be the best. In the placement strategy that is currently
    # winning, intersection, it doesn't actually matter because both lists above (*_qubits) have size 1.
    hor_v, ver_v = None, None
    if horizontal_qubits:
        hor_v = random.choice(horizontal_qubits)
    if vertical_qubits:
        ver_v = random.choice(vertical_qubits)

    if hor_v is None:
        hor_v = (ver_v[0], ver_v[1], 0, random.randint(0, 3))
    if ver_v is None:
        ver_v = (hor_v[0], hor_v[1], 0, random.randint(0, 3))

    return hor_v, ver_v
