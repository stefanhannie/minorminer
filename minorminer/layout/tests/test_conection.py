import random
import unittest

import dwave_networkx as dnx
import minorminer.layout as mml
import networkx as nx


class TestConnection(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestConnection, self).__init__(*args, **kwargs)

        # Graphs for testing
        self.S = nx.random_regular_graph(3, 50)
        self.G = nx.random_regular_graph(3, 150)
        self.C = dnx.chimera_graph(4)
        self.C_coord = dnx.chimera_graph(4, coordinates=True)

        # Layouts for testing
        self.S_layout = mml.p_norm(self.S)
        self.G_layout = mml.p_norm(self.G)
        self.C_layout = mml.dnx_layout(self.C)
        self.C_coord_layout = mml.dnx_layout(self.C_coord)

        # Placements for testing
        self.closest = mml.closest(self.S_layout, self.C_layout)
        self.coord_closest = mml.closest(self.S_layout, self.C_coord_layout)
        self.injective = mml.injective(self.S_layout, self.C_layout)
        self.intersection = mml.intersection(self.S_layout, self.C_layout)
        self.binning = mml.binning(self.S_layout, self.C_layout)
        self.closest_non_dnx = mml.closest(self.S_layout, self.G_layout)

    def test_crosses(self):
        """
        Tests that crosses construction is working correctly.
        """
        # Test different placements
        mml.crosses(self.closest)
        mml.crosses(self.coord_closest)
        mml.crosses(self.injective)
        mml.crosses(self.intersection)
        mml.crosses(self.binning)

        # Only implemented for dnx
        self.assertRaises(NotImplementedError, mml.crosses,
                          self.closest_non_dnx)


if __name__ == '__main__':
    unittest.main()
