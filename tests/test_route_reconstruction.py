import unittest
import networkx as nx
from pathsnap.hmm.route_reconstruction import reconstruct_route


def graph():
    g = nx.MultiDiGraph()
    edges = [(0, 1, 10, 10), (1, 2, 11, 10), (2, 3, 12, 10),
             (3, 4, 13, 10), (1, 2, 99, 20)]
    for u, v, a, d in edges:
        g.add_edge(u, v, key=a, arc_id=a, length_m=d)
    g.graph['arc_endpoints'] = {a: (u, v) for u, v, a, _ in edges}
    return g


class TestRouteReconstruction(unittest.TestCase):
    def test_bridge_and_parallel_arc(self):
        r = reconstruct_route([10, 13], graph())
        self.assertEqual(r.route_arcs, [10, 11, 12, 13])
        self.assertEqual(r.bridges_inserted, 2)
        self.assertEqual(r.breaks, [])

    def test_repeated_and_adjacent(self):
        r = reconstruct_route([10, 10, 11], graph())
        self.assertEqual(r.route_arcs, [10, 11])

    def test_unreachable(self):
        r = reconstruct_route([13, 10], graph())
        self.assertEqual(r.breaks, [(13, 10)])

    def test_cutoff(self):
        r = reconstruct_route([10, 13], graph(), max_bridge_m=15)
        self.assertEqual(r.breaks, [(10, 13)])

    def test_empty(self):
        self.assertEqual(reconstruct_route([], graph()).route_arcs, [])


if __name__ == '__main__':
    unittest.main()
