import unittest
from random import shuffle
from copy import deepcopy
from math import floor, log2

from circuit_constructor import Graph2Cut


class ConstructorTester(unittest.TestCase):

    def graph_constructor(self, node_number, chain_only=False) -> tuple:
        """
        makes a fully connected graph with (at least) all nodes connected in a chain
        then adds different edges to make a graph appear more interconnected than it starts from, looking
        back if there is no repeat among already connected edges.

        makes ~65-68% of possible interconnects in a graph with selected number of nodes

        :param chain_only: a check that forces a graph to only include single chain of all nodes connected
        :return: tuple -> (number of possible edges, list of edges representing graph structure)
        """
        # count possible number of connections
        all_possibilities = sum([*range(1, node_number)])

        # prepare base chain
        node_list = [i for i in range(node_number)]
        shuffle(node_list)
        start_nodes = node_list[:-1]
        end_nodes = node_list[1:]
        starting_edges = [(s_node, e_node) for s_node, e_node in zip(start_nodes, end_nodes)]

        if chain_only:
            return all_possibilities, starting_edges

        # until condition is met, add another couple connections to the graph
        total_edges = deepcopy(starting_edges)
        i = 0
        while len(total_edges) < int(0.68 * all_possibilities):
            # randomly propose a candidate
            shuffle(node_list)
            edge = (node_list[0], node_list[1])
            if edge not in total_edges and edge[::-1] not in total_edges:
                total_edges.append(edge)

            # sanity check - too many random failures
            i += 1
            if i > all_possibilities * 3:
                break

        return all_possibilities, total_edges

    def setUp(self) -> None:
        self.simple_graph_nodes = 3
        self.simple_graph_edges = [(0, 1), (1, 2)]
        self.cuts = 2
        self.end_condition = "="

        self.increasing_size_graphs = [*range(4, 12)]
        self.random_graphs = [(i, self.graph_constructor(i)[1]) for i in self.increasing_size_graphs]
        self.random_chain_graphs = [
            (i, self.graph_constructor(i, chain_only=True)[1])
            for i in self.increasing_size_graphs
        ]

    def test_adder_size(self):
        """
        test adder size allocator function. It should allocate proper number of qbits for any graph

        for example, if graph has 6 edges, a minimum adder should be able to add to 2^3 -> 8
        so size should be 3 qbits (2^0, 2^1, 2^2 -> in total 1 + 2 + 4)

        another example 47 edges -> "63 adder" (6 bit -> 1 + 2 + 4 + 8 + 16 + 32
        """
        test_circuit_builder = Graph2Cut(
            self.simple_graph_nodes, self.simple_graph_edges,
            self.cuts, self.end_condition
        )
        test_circuit_builder._allocate_qbits()
        self.assertEqual(len(test_circuit_builder.quantum_adder_register), 2)

        for graph in self.random_graphs:
            test_circuit_builder2 = Graph2Cut(
                graph[0], graph[1],
                cuts_number=len(graph[1]), condition=self.end_condition
            )
            test_circuit_builder2._allocate_qbits()
            if 2 <= len(graph[1]) < 4:
                self.assertEqual(len(test_circuit_builder2.quantum_adder_register), 2)
            elif 4 <= len(graph[1]) < 8:
                self.assertEqual(len(test_circuit_builder2.quantum_adder_register), 3)
            elif 8 <= len(graph[1]) < 16:
                self.assertEqual(len(test_circuit_builder2.quantum_adder_register), 4)
            elif 16 <= len(graph[1]) < 32:
                self.assertEqual(len(test_circuit_builder2.quantum_adder_register), 5)
            elif 32 <= len(graph[1]) < 64:
                self.assertEqual(len(test_circuit_builder2.quantum_adder_register), 6)

    def test_simple_qbit_allocation(self):
        """
        test qbit/cbit registers being properly allocated for a graph structure that
        only consist a single chain of edges (no other interconnections)
        """
        for graph in self.random_chain_graphs:
            nodes, edges = graph
            test_circuit_builder = Graph2Cut(nodes, edge_list=edges, cuts_number=len(edges), condition='=')
            test_circuit_builder._allocate_qbits()
            self.assertEqual(len(test_circuit_builder.quantum_adder_register), floor(log2(nodes-1))+1)
            self.assertEqual(len(test_circuit_builder.edge_qbit_register), len(edges))
            self.assertEqual(len(test_circuit_builder.results_register), nodes)
            self.assertEqual(len(test_circuit_builder.node_qbit_register), nodes)
            self.assertEqual(len(test_circuit_builder.ancilla_qbit_register), 1)


if __name__ == '__main__':
    # for i in range(4, 15):
    #     possibilities, edges_chosen = graph_constructor(i)
    #     print(i, possibilities, f"edge_statistic {len(edges_chosen) / possibilities}")
    #     print(len(edges_chosen), edges_chosen[:int(len(edges_chosen) * 0.75)], "...")
    unittest.main()

