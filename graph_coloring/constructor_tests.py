import unittest
from random import shuffle, choice, seed
from copy import deepcopy
from math import floor, log2
from time import time

from circuit_constructor import Graph2Cut
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit_aer.backends import AerSimulator


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

        # computational complexity for a naive approach for 12 node graph is already huge, stopping at this level
        self.increasing_size_graphs = [*range(4, 13)]
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

        another example 47 edges -> "63 adder" (6 bit -> 1 + 2 + 4 + 8 + 16 + 32)
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

    def test_edge_detector_chain_case(self):
        """
        test whether edges are properly detected and flagged as '1' in respected qbits
        """
        # here graph data does matter, though only first few steps of algorithm are checked
        # 1 - node color mismatch -> a cut; 0 - node color matching -> no cut

        # special case - chain graph, coloring case 01010101010101... -> all edges = 1
        # example seed: seed_ = 1700207863.558291
        # chain order -> [(1, 6), (6, 7), (7, 0), (0, 4), (4, 5), (5, 3), (3, 2)]
        # initialized state -> "01011001"
        # simulation result -> {"1111111": 10}

        seed_ = time()
        seed(seed_)
        print(f"time seed for edge_detector test for chain graph: {seed_}")
        nodes = 8
        _, chain_graph = self.graph_constructor(nodes, chain_only=True)
        graph_cutter = Graph2Cut(nodes=nodes, edge_list=chain_graph, cuts_number=len(chain_graph))
        test_circuit_book = graph_cutter.assemble_subcircuits()
        edge_checker_subcircuit = test_circuit_book[0]
        c_register = ClassicalRegister(graph_cutter.edge_qbit_register.size, name="test_register")

        # force all edges to be detected as 1 by careful initialization
        starting_index = chain_graph[0][0]
        node_chain = [starting_index] + [e[1] for e in chain_graph]
        node_values = [
            (node_index, 1 if i % 2 == 0 else 0)
            for i, node_index in enumerate(node_chain)
        ]

        test_circuit = QuantumCircuit(graph_cutter.node_qbit_register, graph_cutter.edge_qbit_register, c_register)
        for node_index, qbit_starting_value in node_values:
            if qbit_starting_value:
                test_circuit.x(graph_cutter.node_qbit_register[node_index])

        measurement_circuit = QuantumCircuit(graph_cutter.edge_qbit_register, c_register)
        measurement_circuit.measure(graph_cutter.edge_qbit_register, c_register)

        test_circuit.compose(edge_checker_subcircuit, inplace=True)
        test_circuit.compose(measurement_circuit, graph_cutter.edge_qbit_register, inplace=True)
        simulator = AerSimulator()
        job = simulator.run(test_circuit, shots=10)
        counts = job.result().get_counts()
        counts_list = [(measurement, counts[measurement]) for measurement in counts]
        self.assertEqual(counts_list[0][0], "".join(['1' for _ in range(graph_cutter.edge_qbit_register.size)]))
        self.assertEqual(counts_list[0][1], 10)

    @unittest.skip("not implemented")
    def test_adder(self):
        """
        test if adder sub-circuit functions and counts properly qbits that are given for certain grover problem
        """
        # here we just use
        # 1 - node color mismatch -> a cut; 0 - node color matching -> no cut
        seed_ = time()
        seed(seed_)
        print(f"time seed for test_adder: {seed_}")
        random_graph = choice(self.random_graphs)
        nodes, edges = random_graph
        two_cut = Graph2Cut(nodes=nodes, edge_list=edges, cuts_number=len(edges))
        sub_circuit_book = two_cut.assemble_subcircuits()
        edge_checker_circuit = sub_circuit_book[0]
        adder_circuit = sub_circuit_book[1]

        # generate 'simulated cut occurrences' and initialize them as "X" gates in edge register
        test_cases = [
            "".join(['0' for _ in range(two_cut.edge_qbit_register.size)]),  # special case - 0
            "".join(['1' for _ in range(two_cut.edge_qbit_register.size)]),  # special case - full 1 state
            # random 10 cases that should
            *["".join([choice(["0", "1"]) for _ in range(two_cut.edge_qbit_register.size)]) for __ in range(10)]
        ]

        for t in test_cases:
            correct_count = sum([int(c == "1") for c in t])
            # prepare full circuit for future composition purposes
            c_register = ClassicalRegister(two_cut.quantum_adder_register.size, name="test_register")
            test_circuit = QuantumCircuit(two_cut.edge_qbit_register, two_cut.quantum_adder_register, c_register)

            # prepare measurement operation
            measurement_circuit = QuantumCircuit(two_cut.quantum_adder_register, c_register)
            measurement_circuit.measure(two_cut.quantum_adder_register, c_register)

    @unittest.skip("not implemented")
    def test_flag_by_condition(self):
        """
        test if condition sub-circuit really flags the state for given solution
        """

    @unittest.skip("not implemented")
    def test_grover_diffusion(self):
        """
        test if 'inversion by the mean' really works for given solutions
        """

    @unittest.skip("not implemented")
    def test_chains(self):
        """
        test circuit creations and functionality for special case of only 2 solutions being proper in given condition

        there are chains given in this task:
        ...-o-o-o-o-o-o-o-o-...
        which yield only 2 possible solutions of max bipartiteness, being:
        ...-1-0-1-0-1-0-1-0-... and ...-0-1-0-1-0-1-0-1-...
        meaning solutions have to compliment each other and being a sequences of interleaved 0's and 1's
        """

    @unittest.skip("not implemented")
    def test_simple_rings(self):
        """
        test circuit creations and functionality for a case of rings

        there will be 2 possibilities for maximum bipartiteness, being:
        (these are only for illustrative purposes and only present part of the ring with idea)
            o-o            o-o-o
           /   \          /     \
          o     o        o       o
          \   ...         \   ...
           o-o             o-o
        meaning - there can be odd number of nodes and even number of nodes

        odd number of nodes will not yield a full cut in 2 color cases
        even number of nodes will yield full cut solution similar to a chain case

        odd number of nodes will yield a couple of mirrored solutions if we allow 1 edge not to be cut
        even number of nodes will yield a lot of solutions if we allow 1 edge not to be cut

        based on this we could create a test to check if automated circuit creation works properly
        """

    @unittest.skip("not implemented")
    def test_random_graph(self):
        """
        test a random fully connected graph, for its bipartiteness in a given condition
        look for the most probable answer and check if it satisfies solution condition

        for this test, a new method of graph creation has to be created to grant 100% chance of getting
        at least 1 proper solution
        """


if __name__ == '__main__':
    # for i in range(4, 15):
    #     possibilities, edges_chosen = graph_constructor(i)
    #     print(i, possibilities, f"edge_statistic {len(edges_chosen) / possibilities}")
    #     print(len(edges_chosen), edges_chosen[:int(len(edges_chosen) * 0.75)], "...")
    unittest.main()
