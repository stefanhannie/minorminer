import random


def random_remove(placement, percent=2/3, **kwargs):
    """
    Given a placement (a map, phi, from vertices of S to the subsets of vertices of T), randomly remove the given 
    percent of vertices from each chain.

    Parameters
    ----------
    placement : placement.Placement
        A mapping from vertices of S (keys) to subsets of vertices of T (values).
    percent : float (default 2/3)
        If True, gets the closed 2nd neighborhood of each vertex. If False, get the closed 1st neighborhood of each
        vertex.

    Returns
    -------
    placement : placement.Placement
        A mapping from vertices of S (keys) to subsets of vertices of T (values).
    """
    for v, C in placement.items():
        chain_list = list(C)  # In case C is a set/frozenset or something

        # Shuffle and remove some qubits
        random.shuffle(chain_list)
        for _ in range(int(len(C)*percent)):
            chain_list.pop()

        # Update the chains
        placement[v] = chain_list

    return placement
