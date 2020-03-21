import random
import warnings
from collections import abc, Counter, defaultdict
from itertools import cycle, product

import networkx as nx
import numpy as np
from . import layout
from .utils import (dnx_utils, graph_utils, layout_utils,
                    placement_utils)
from scipy.spatial import KDTree, distance


def closest(S_layout, T_layout, fill_T=False, **kwargs):
    """
    Top level function for minorminer.layout.__init__() use as a parameter.
    # FIXME: There's surely a better way of doing this.
    """
    P = Placement(S_layout, T_layout, fill_T=fill_T)
    _ = P.closest(**kwargs)
    return P


def injective(S_layout, T_layout, fill_T=False, **kwargs):
    """
    Top level function for minorminer.layout.__init__() use as a parameter.
    # FIXME: There's surely a better way of doing this.
    """
    P = Placement(S_layout, T_layout, fill_T=fill_T)
    _ = P.injective(**kwargs)
    return P


def intersection(S_layout, T_layout, fill_T=False, **kwargs):
    """
    Top level function for minorminer.layout.__init__() use as a parameter.
    # FIXME: There's surely a better way of doing this.
    """
    P = Placement(S_layout, T_layout, fill_T=fill_T)
    _ = P.intersection(**kwargs)
    return P


def binning(S_layout, T_layout, fill_T=False, **kwargs):
    """
    Top level function for minorminer.layout.__init__() use as a parameter.
    # FIXME: There's surely a better way of doing this.
    """
    P = Placement(S_layout, T_layout, fill_T=fill_T)
    _ = P.binning(**kwargs)
    return P


class Placement(abc.MutableMapping):
    def __init__(
        self,
        S_layout,
        T_layout,
        placement=None,
        fill_T=False,
        **kwargs
    ):
        """
        Compute a placement of S in T, i.e., map V(S) to 2^{V(T)}.

        Parameters
        ----------
        S_layout : layout.Layout or networkx.Graph
            A layout for S; i.e. a map from S to R^d. Or a networkx graph to make a layout from.
        T_layout : layout.Layout or networkx.Graph
            A layout for T; i.e. a map from T to R^d. Or a networkx graph to make a layout from.
        placement : dict (default None)
            You can specify a pre-computed placement for S in T.
        fill_T : bool (default False)
            If True, S_layout is scaled to the scale of T_layout. If False, S_layout uses its scale.
        kwargs : dict
            Keyword arguments are passed to one of the placement algorithms below.
        """
        self.S_layout = placement_utils.parse_layout(S_layout)
        self.T_layout = placement_utils.parse_layout(T_layout)

        # Layout dimensions should match
        if self.S_layout.d != self.T_layout.d:
            raise ValueError(
                "S_layout has dimension {} but T_layout has dimension {}. These must match.".format(
                    self.S_layout.d, self.T_layout.d)
            )

        # Extract the graphs
        self.S = self.S_layout.G
        self.T = self.T_layout.G

        # Scale S if the user wants to, or if S_layout is bigger than T_layout
        if fill_T or np.any(np.abs(self.S_layout.layout_array) > self.T_layout.scale):
            self.fill_T = True
            self.S_layout.scale_layout(self.T_layout.scale)
        else:
            self.fill_T = False

        self.placement = placement

    @property
    def placement(self):
        return self._placement

    @placement.setter
    def placement(self, value):
        self._placement = value
        self.chains = placement_utils.convert_to_chains(value)

    # The layout class should behave like a dictionary
    def __iter__(self):
        """
        Iterate through the keys of the dictionary chains.
        """
        yield from self.chains

    def __getitem__(self, key):
        """
        Get the chain value at the key vertex.
        """
        return self.chains[key]

    def __setitem__(self, key, value):
        """
        Set the chain value at the key vertex.
        """
        self.chains[key] = value

    def __delitem__(self, key):
        """
        Delete the chain value at the key vertex.
        """
        del self.chains[key]

    def __repr__(self):
        """
        Use the chain's dictionary representation.
        """
        return repr(self.chains)

    def __len__(self):
        """
        The length of a placement is the length of the chains dictionary.
        """
        return len(self.chains)

    def closest(self, subset_size=(1, 1), num_neighbors=1, **kwargs):
        """
        Maps vertices of S to the closest vertices of T as given by S_layout and T_layout. i.e. For each vertex u in
        S_layout and each vertex v in T_layout, map u to the v with minimum Euclidean distance (||u - v||_2).

        Parameters
        ----------
        subset_size : tuple (default (1, 1))
            A lower and upper bound on the size of subets of T that will be considered when mapping vertices of S.
        num_neighbors : int (default 1)
            The number of closest neighbors to query from the KDTree--the neighbor with minimium overlap is chosen.

        Returns
        -------
        placement : dict
            A mapping from vertices of S (keys) to subsets of vertices of T (values).
        """
        # A new layout for subsets of T.
        T_subgraph_layout = {}

        # Get connected subgraphs to consider mapping to
        T_subgraphs = graph_utils.get_connected_subgraphs(
            self.T, subset_size[0], subset_size[1])

        # Calculate the barycenter (centroid) of each subset
        for k in range(subset_size[0], subset_size[1]+1):
            if k == 1:
                for subgraph in T_subgraphs[k]:
                    v, = subgraph  # Iterable unpacking
                    T_subgraph_layout[subgraph] = self.T_layout[v]
            else:
                for subgraph in T_subgraphs[k]:
                    T_subgraph_layout[subgraph] = np.mean(
                        np.array([self.T_layout[v] for v in subgraph]), axis=0)

        # Use scipy's KDTree to solve the nearest neighbor problem.
        # This requires a few lookup tables
        T_vertex_lookup = {tuple(p): V for V, p in T_subgraph_layout.items()}
        layout_points = [tuple(p) for p in T_subgraph_layout.values()]
        # print(layout_points)
        overlap_counter = Counter()
        tree = KDTree(layout_points)

        placement = {}
        for u, u_pos in self.S_layout.items():
            distances, v_indices = tree.query(u_pos, num_neighbors)
            placement[u] = placement_utils.minimize_overlap(
                distances, v_indices, T_vertex_lookup, layout_points, overlap_counter)

        self.placement = placement
        return self.placement

    def injective(self, **kwargs):
        """
        Injectively maps vertices of S to the closest vertices of T as given by S_layout and T_layout. This is the
        assignment problem. To solve this it builds a complete bipartite graph between S and T with edge weights the
        Euclidean distances of the incident vertices; a minimum weight full matching is then computed. This runs in
        O(|S||T|log(|T|)) time.

        Returns
        -------
        placement : dict
            A mapping from vertices of S (keys) to vertices of T (values).
        """
        # Relabel the vertices from S and T in case of name conflict; S --> 0 and T --> 1.
        X = nx.Graph()
        X.add_edges_from(
            (
                ((0, u), (1, v), dict(weight=distance.euclidean(u_pos, v_pos)))
                for (u, u_pos), (v, v_pos) in product(self.S_layout.items(), self.T_layout.items())
            )
        )
        M = nx.bipartite.minimum_weight_full_matching(
            X, ((0, u) for u in self.S_layout))

        self.placement = {u: [M[(0, u)][1]] for u in self.S_layout}
        return self.placement

    def intersection(self, **kwargs):
        """
        Map each vertex of S to its nearest row/column intersection qubit in T (T must be a D-Wave hardware graph). 

        Returns
        -------
        placement : dict
            A mapping from vertices of S (keys) to vertices of T (values).
        """
        # Currently only implemented for 2d chimera
        if self.T.graph.get("family") != "chimera" or self.T_layout.d != 2:
            raise NotImplementedError(
                "This strategy is currently only implemented for Chimera in 2d.")

        # Bin vertices of S and T into a grid graph G
        G = self._intersection_binning()

        placement = {}
        for _, data in G.nodes(data=True):
            for v in data["variables"]:
                placement[v] = data["qubits"]

        self.placement = placement
        return self.placement

    def _intersection_binning(self):
        """
        Map the vertices of S to the "intersection graph" of T. This modifies the grid graph G by assigning vertices from S 
        and T to vertices of G.

        Returns
        -------
        G : networkx.Graph
            A grid graph. Each vertex of G contains data attributes "variables" and "qubits", that is, respectively 
            vertices of S and T assigned to that vertex.  
        """
        # Scale the layout so that for each vertical and horizontal qubit that cross each other, we have an integer point.
        m, n, t = dnx_utils.lookup_dnx_dims(self.T)

        # Make the "intersection graph" of the dnx_graph
        # Grid points correspond to intersection rows and columns of the dnx_graph
        G = nx.grid_2d_graph(t*m, t*n)

        # Determine the scale for putting things in the positive quadrant
        scale = (t*max(n, m)-1)/2

        # Get the row, column mappings for the dnx graph
        lattice_mapping = dnx_utils.lookup_intersection_coordinates(
            self.T)

        # Less efficient, but more readable to initialize all at once
        for v in G:
            G.nodes[v]["qubits"] = set()
            G.nodes[v]["variables"] = set()

        # Add qubits (vertices of T) to grid points
        for int_point, Q in lattice_mapping.items():
            G.nodes[int_point]["qubits"] |= Q

        # --- Map the S_layout to the grid
        # D-Wave counts the y direction like matrix rows; inversion makes pictures match
        modified_layout = layout.invert_layout(
            self.S_layout.layout_array, self.S_layout.center)

        # "Zoom in" on layout_S so that the integer points are better represented
        if self.fill_T:
            zoom_scale = scale
        else:
            zoom_scale = t*self.S_layout.scale
        modified_layout = layout.scale_layout(
            modified_layout, zoom_scale, self.S_layout.scale, self.S_layout.center)

        # Center to the positive orthant
        modified_layout = layout.center_layout(
            modified_layout, 2*(scale,), self.S_layout.center)

        # Turn it into a dictionary
        modified_layout = {v: pos for v, pos in zip(
            self.S, modified_layout)}

        # Add "variables" (vertices from S) to grid points too
        for v, pos in modified_layout.items():
            grid_point = tuple(int(x) for x in np.round(pos))
            G.nodes[grid_point]["variables"].add(v)

        return G

    def binning(self, unit_tile_capacity=None, strategy="layout", **kwargs):
        """
        Map the vertices of S to the vertices of T by first mapping both to an integer lattice (T must be a D-Wave hardware graph). 

        Parameters
        ----------
        unit_tile_capacity : int (default None)
            The number of variables (vertices of S) that are allowed to map to unit tiles of T. If set, a topple based algorithm is run
            to ensure that not too many variables are contained in the same unit tile of T.
        strategy : str (default "layout")
            layout : Use S_layout to determine the mapping from variables to qubits.
            cycle : Cycle through the variables and qubits in a unit cell and assign variables to qubits one at a time, repeating if necessary.
            all : Map each variable to each qubit in a unit cell. Lots of overlap.

        Returns
        -------
        placement : dict
            A mapping from vertices of S (keys) to vertices of T (values).
        """
        # Currently only implemented for 2d chimera
        if self.T.graph.get("family") != "chimera" or self.T_layout.d != 2:
            raise NotImplementedError(
                "This strategy is currently only implemented for Chimera in 2d.")

        # Get the lattice point mapping for the dnx graph
        m, n, t = dnx_utils.lookup_dnx_dims(self.T)

        # Make the grid "quotient" of the dnx_graph
        # Quotient the ~K_4,4 unit cells of the dnx_graph to grid points
        G = nx.grid_2d_graph(m, n)

        # Determine the scale for putting things in the positive quadrant
        scale = (max(n, m)-1)/2

        # Get the grid graph and the modified layout for S
        modified_S_layout = self._unit_cell_binning(G, scale)

        # Do we need to topple?
        if unit_tile_capacity or strategy == "layout":
            unit_tile_capacity = unit_tile_capacity or t
            n, N = len(self.S_layout), m*n*unit_tile_capacity
            if n > N:
                raise RuntimeError(
                    "You're trying to fit {} vertices of S into {} spots of T.".format(n, N))
            _topple(G, modified_S_layout, unit_tile_capacity)

        # Build the placement
        placement = defaultdict(set)
        if strategy == "layout":
            for g, V in G.nodes(data="variables"):
                V = list(V)

                x_indices, y_indices = list(range(t)), list(range(t))
                for _ in range(t, len(V), -1):
                    x_indices.remove(random.choice(x_indices))
                    y_indices.remove(random.choice(y_indices))

                # Run through the sorted points and assign them to qubits--find a transveral in each unit cell.
                for k in np.argsort([modified_S_layout[v] for v in V], 0):
                    # The x and y order in the argsort (k_* in [0,1,2,3])
                    k_x, k_y = k[0], k[1]
                    # The vertices of S at those indicies
                    u_x, u_y = V[k_x], V[k_y]
                    placement[u_x].add((g[1], g[0], 0, x_indices[k_x]))
                    placement[u_y].add((g[1], g[0], 1, y_indices[k_y]))
            placement = dnx_utils.relabel_chains(self.T, placement)

        if strategy == "cycle":
            for _, data in G.nodes(data=True):
                if data["variables"]:
                    for v, q in zip(data["variables"], cycle(data["qubits"])):
                        placement[v].add(q)

        elif strategy == "all":
            for _, data in G.nodes(data=True):
                if data["variables"]:
                    for v in data["variables"]:
                        placement[v] |= data["qubits"]

        self.placement = placement
        return self.placement

    def _unit_cell_binning(self, G, scale):
        """
        Map the vertices of S to the unit cell quotient of T. This modifies the grid graph G by assigning vertices from S 
        and T to vertices of G.

        Parameters
        ----------
        G : networkx.Graph
            A grid_2d_graph representing the lattice points in the positive quadrant.
        scale : float
            The scale necessary to translate (and/or resize) the layouts so that they occupy the positive quadrant.

        Returns
        -------
        modified_layout : dict
            The layout of S after translating and scaling to the positive quadrant. 
        """
        # Get the lattice point mapping for the dnx graph
        lattice_mapping = dnx_utils.lookup_grid_coordinates(self.T)

        # Less efficient, but more readable to initialize all at once
        for v in G:
            G.nodes[v]["qubits"] = set()
            G.nodes[v]["variables"] = set()

        # Add qubits (vertices of T) to grid points
        for v, int_point in lattice_mapping.items():
            G.nodes[int_point]["qubits"].add(v)

        # --- Map the S_layout to the grid
        # D-Wave counts the y direction like matrix rows; inversion makes pictures match
        modified_layout = layout.invert_layout(
            self.S_layout.layout_array, self.S_layout.center)

        # Scale S to fill T at the grid level
        if self.fill_T:
            modified_layout = layout.scale_layout(
                modified_layout, scale, self.S_layout.scale, self.S_layout.center)

        # Center to the positive orthant
        modified_layout = layout.center_layout(
            modified_layout, 2*(scale,), self.S_layout.center)

        # Turn it into a dictionary
        modified_layout = {v: pos for v, pos in zip(
            self.S, modified_layout)}

        # Add "variables" (vertices from S) to grid points too
        for v, pos in modified_layout.items():
            grid_point = tuple(int(x) for x in np.round(pos))
            G.nodes[grid_point]["variables"].add(v)

        return modified_layout


def _topple(G, modified_layout, unit_tile_capacity):
    """
    Modifies G by toppling.

    topple : Topple grid points so that the number of variables at each point does not exceed specified unit_tile_capacity.
            After toppling assign the vertices of S to a transversal of qubits at the grid point.
    """
    # Check who needs to move (at most unit_tile_capacity vertices of S allowed per grid point)
    moves = {v: 0 for _, V in G.nodes(data="variables") for v in V}
    topple = True  # This flag is set to true if a topple happened this round
    stop = 0
    while topple:
        stop += 1
        topple = False
        for g, V in G.nodes(data="variables"):
            num_vars = len(V)

            # If you're too full let's topple/chip fire/sand pile
            if num_vars > unit_tile_capacity:
                topple = True

                # Which neighbor do you send it to?
                neighbors_capacity = {
                    v: len(G.nodes[v]["variables"]) for v in G[g]}

                # Who's closest?
                positions = {v: modified_layout[v]
                             for v in G.nodes[g]["variables"]}

                while num_vars > unit_tile_capacity:
                    # The neighbor to send it to
                    lowest_occupancy = min(neighbors_capacity.values())
                    hungriest = random.choice(
                        [v for v, cap in neighbors_capacity.items() if cap == lowest_occupancy])

                    # Who to send
                    min_score = float("inf")
                    for v, pos in positions.items():
                        dist = np.linalg.norm(np.array(hungriest) - pos)
                        score = dist + moves[v]
                        if score < min_score:
                            min_score = min(score, min_score)
                            moves[v] += 1
                            food = v

                    G.nodes[g]["variables"].remove(food)
                    del positions[food]
                    G.nodes[hungriest]["variables"].add(food)

                    neighbors_capacity[hungriest] += 1
                    num_vars -= 1

        if stop == 1000:
            raise RuntimeError(
                "I couldn't topple, this may be an infinite loop.")
