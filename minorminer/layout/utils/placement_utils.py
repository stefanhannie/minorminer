import networkx as nx
import numpy as np

from ..layout import Layout, dnx_layout, p_norm


def parse_layout(G):
    """
    Takes in a layout.Layout, a networkx.Graph, or a dict and returns a layout.Layout object.
    """
    if isinstance(G, Layout):
        return Layout(G.G, G.layout)

    if isinstance(G, nx.Graph):
        if G.graph.get("family") in ("chimera", "pegasus"):
            return dnx_layout(G)
        else:
            return p_norm(G)

    if isinstance(G, dict):
        raise TypeError(
            "If you want to pass in a precomputed layout dictionary, please create a layout object; Layout(G, layout).")


def minimize_overlap(distances, v_indices, T_vertex_lookup, layout_points, overlap_counter):
    """
    A greedy penalty-type model for choosing overlapping chains.
    """
    # KDTree.query either returns a single index or a list of indexes depending on how many neighbors are queried.
    if isinstance(v_indices, np.int64):
        return T_vertex_lookup[layout_points[v_indices]]

    subsets = {}
    for i in v_indices:
        subset = T_vertex_lookup[layout_points[i]]
        subsets[subset] = sum(d + 10**overlap_counter[v]
                              for d, v in zip(distances, subset))

    cheapest_subset = min(subsets, key=subsets.get)
    overlap_counter.update(cheapest_subset)
    return cheapest_subset


def convert_to_chains(placement):
    """
    Helper function to convert a placement to a chain-ready data structure.
    """
    if placement is None:
        return {}
    for v in placement.values():
        if isinstance(v, (list, frozenset, set)):
            return dict(placement)
        return {v: [q] for v, q in placement.items()}
