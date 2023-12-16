import unittest
from collections import OrderedDict
from itertools import pairwise
# from pprint import pprint
from random import shuffle, choice, seed, randint
from copy import deepcopy
from math import floor, log2, sqrt, pi
from time import time

# from qiskit.circuit import CircuitInstruction

from circuit_constructor import Graph2Cut
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit_aer.backends import AerSimulator


class ConstructorTester(unittest.TestCase):

    @staticmethod
    def instruction_usage_qbits(graph_cutter: Graph2Cut, edge: tuple[int, ...], edge_index: int):
        """
        For adder purposes, construct possible tuples with qubits that need to be used for each gate in given circuit
        instruction.

        :param graph_cutter: graph class that contains qubits to be checked
        :param edge: tuple with node initials that should form an edge detection check
        :param edge_index: index that informs about current size of the adder pyramid to be constructed
        :return: list of tuples with Qubit objects serving as a check
        """

        # init and node qubit usage correctness check
        qubit_gate_usage_list = []
        for _ in range(2):
            qubit_gate_usage_list.append(tuple([
                graph_cutter.node_qbit_register[edge[0]],
                graph_cutter.ancilla_qbit_register[0]
            ]))
            qubit_gate_usage_list.append(tuple([
                graph_cutter.node_qbit_register[edge[1]],
                graph_cutter.ancilla_qbit_register[0]
            ]))

        # pyramid qbit usage correctness check
        # print(f"number of gates for the {edge_index} = {floor(log2(edge_index+1))+1}")
        for i in range(floor(log2(edge_index+1))+1):
            # print(f"adder gate nr {i} for edge {edge_index}")
            qubit_gate_usage_list.append(tuple([
                graph_cutter.ancilla_qbit_register[0],
                *graph_cutter.quantum_adder_register[0:i+1]
            ]))

        return qubit_gate_usage_list

    @staticmethod
    def graph_constructor(node_number, chain_only=False) -> tuple:
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
            solution_node_sequence = [
                (edge[0], "0" if index % 2 == 0 else "1") for index, edge in enumerate(starting_edges)
            ]
            solution_node_sequence += [(starting_edges[-1][1], "0" if solution_node_sequence[-1][1] == "1" else "1")]
            solution = "".join([n[1] for n in sorted(solution_node_sequence, key=lambda x: x[0])])
            return starting_edges, solution[::-1]

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

        return total_edges, None

    @staticmethod
    def graph_with_certain_solution(node_number) -> tuple:
        """
        take a number of nodes, mix them up into 2 piles, colour them and connect in edges to make a final
        graph with at least 1 valid solution

        then throw in a couple additional valid edges to test the solution for full graph split

        :param node_number: a total number of nodes to be in a graph structure
        :return: tuple -> (valid solution -> string of 0's and 1's which should appear among solutions,
            list of edges representing graph structure)
        """
        all_possibilities = sum([*range(1, node_number)])
        nodes = [i for i in range(node_number)]
        shuffle(nodes)
        red = nodes[:int(len(nodes) / 2)]
        blue = nodes[int(len(nodes) / 2):]
        blue_copy = deepcopy(blue)
        red_copy = deepcopy(red)

        # take turns taking numbers from red and blue piles, to form a chain that is a valid full partition
        # always start from bigger list (assume blue), swap if necessary
        if len(red_copy) > len(blue_copy):
            red_copy, blue_copy = blue_copy, red_copy

        valid_solution = []
        interleaved_chain = []
        for i in range(len(nodes)):
            if i % 2 == 0:
                element = blue_copy.pop()
                interleaved_chain.append(element)
                valid_solution.append((element, "0"))
            else:
                element = red_copy.pop()
                interleaved_chain.append(element)
                valid_solution.append((element, "1"))

        valid_solution_str = "".join([e[1] for e in sorted(valid_solution, key=lambda x: x[0])])
        list_of_edges = [e for e in pairwise(interleaved_chain)]

        # append another edge or couple more to make graph interconnected further
        total_edges = deepcopy(list_of_edges)
        i = 0
        while len(total_edges) < int(0.68 * all_possibilities):
            # find 2 nodes of opposing colors
            blue_node = choice(blue)
            red_node = choice(red)

            edge = (red_node, blue_node)
            if edge not in total_edges and edge[::-1] not in total_edges:
                total_edges.append(edge)

            # sanity check - too many random failures
            i += 1
            if i > all_possibilities:
                break

        return valid_solution_str, total_edges

    def prepare_circuit_measurements(self, tested_circuit: QuantumCircuit, shots=10):
        """perform circuit simulation from tested circuit"""
        # simulation methods = ('automatic', 'statevector', 'density_matrix', 'stabilizer',
        # 'matrix_product_state', 'extended_stabilizer', 'unitary', 'superop')
        job = AerSimulator().run(tested_circuit, shots=shots, method='automatic')
        counts = job.result().get_counts()
        counts_list = [(measurement, counts[measurement]) for measurement in counts]
        return counts_list

    def setUp(self) -> None:
        self.simple_graph_nodes = 3
        self.simple_graph_edges = [(0, 1), (1, 2)]
        self.cuts = 2
        self.end_condition = "="

        # computational complexity for a naive approach for 12 node graph is already huge, stopping at this level
        self.increasing_size_graphs = [*range(4, 13)]
        self.random_graphs = [(i, self.graph_constructor(i)[0]) for i in self.increasing_size_graphs]
        self.random_chain_graphs = [
            (i, *self.graph_constructor(i, chain_only=True))
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
        # print(f"time seed for edge_detector test for chain graph: {seed_}")
        nodes = 8
        chain_graph, __ = self.graph_constructor(nodes, chain_only=True)
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

        counts_list = self.prepare_circuit_measurements(test_circuit)
        self.assertEqual(counts_list[0][0], "".join(['1' for _ in range(graph_cutter.edge_qbit_register.size)]))
        self.assertEqual(counts_list[0][1], 10)

    def test_adder(self):
        """
        test if regular adder sub-circuit functions and counts properly qbits that are given for certain grover problem

        this is very resource extensive version (memory), thus we use only a small graph to check the adder
        """
        # here we just use
        # 1 - node color mismatch -> a cut; 0 - node color matching -> no cut
        # seed_ = time()
        seed_ = 1700216428.893547
        seed(seed_)
        # print(f"time seed for test_adder: {seed_}")
        random_graph = self.random_graphs[4]

        nodes, edges = random_graph
        graph_cutter = Graph2Cut(nodes=nodes, edge_list=edges, cuts_number=len(edges))
        sub_circuit_book = graph_cutter.assemble_subcircuits()
        adder_circuit = sub_circuit_book[1]

        # generate 'simulated cut occurrences' and initialize them as "X" gates in edge register
        test_cases = [
            "".join(['0' for _ in range(graph_cutter.edge_qbit_register.size)]),  # special case - 0
            "".join(['1' for _ in range(graph_cutter.edge_qbit_register.size)]),  # special case - full 1 state
            # random 10 cases that should simulate some random inputs into the graph
            *["".join([choice(["0", "1"]) for _ in range(graph_cutter.edge_qbit_register.size)]) for __ in range(10)]
        ]
        # print(test_cases)

        for case in test_cases:
            correct_count = sum([int(c == "1") for c in case])
            # prepare full circuit for future composition purposes
            c_register = ClassicalRegister(graph_cutter.quantum_adder_register.size, name="test_register")
            test_circuit = QuantumCircuit(
                graph_cutter.edge_qbit_register, graph_cutter.quantum_adder_register, c_register)

            for node_index, qbit_starting_value in enumerate(case):
                if int(qbit_starting_value):
                    test_circuit.x(graph_cutter.edge_qbit_register[node_index])
            test_circuit.barrier()

            # prepare measurement operation
            measurement_circuit = QuantumCircuit(graph_cutter.quantum_adder_register, c_register)
            measurement_circuit.measure(graph_cutter.quantum_adder_register, c_register)

            test_circuit.compose(adder_circuit, inplace=True)
            test_circuit.compose(measurement_circuit, graph_cutter.quantum_adder_register, inplace=True)
            counts_list = self.prepare_circuit_measurements(test_circuit)
            # print(counts_list)
            # break
            # print(case, counts_list)
            self.assertEqual(correct_count, int(counts_list[0][0], base=2))
            self.assertEqual(10, counts_list[0][1])

    def test_qbit_reuse_adder_parts(self):
        """
        test sub-circuit creating of the qbit reuse adder

        first a subtest is prepared to check if Qubit ordering works properly and proper Qubits are used to make
        certain sub-circuits, then the real addition is checked (for each sub-circuit created)
        """
        seed_ = time()
        # seed_ = 1700216428.893547
        seed(seed_)
        # print(f"time seed for test_qbit_reuse_adder: {seed_}")
        random_graph = self.random_chain_graphs[4]

        nodes, edges = random_graph
        graph_cutter = Graph2Cut(nodes=nodes, edge_list=edges, cuts_number=len(edges), optimization='qbits')
        sub_circuit_book = graph_cutter.assemble_subcircuits()
        edge_counting_subcircuits: list[QuantumCircuit] = sub_circuit_book[0]
        test_cases = ["00", "01", "10", "11"]

        self.assertEqual(len(edges), len(edge_counting_subcircuits))

        # structure of any sub-circuit in this solution should be - check, addition_pyramid, check_reverse
        # entire array of sub-circuits should make up for an addition mechanism that uses only a single ancilla and no
        # color-mismatch qbit storage would be needed making up a qbit-savings

        # construct a list of tuples which represent qbits used in each gate that is a part of every sub-circuit
        # when its made. This way we make sure that gates are made up from correct pieces within sub-circuits
        for edge_index, (subcircuit, edge) in enumerate(zip(edge_counting_subcircuits, edges)):
            all_ops: OrderedDict = subcircuit.count_ops()
            # print(f'\nnext sub-circuit: {edge}, {edge_index}')
            predicted_instruction_subregisters = self.instruction_usage_qbits(graph_cutter, edge, edge_index)
            actual_subregisters = []
            for op in all_ops:
                actual_subregisters.extend([instruction.qubits for instruction in subcircuit.get_instructions(op)])

            for qubits_tuple in deepcopy(predicted_instruction_subregisters):
                # print(f"currently checked tup: {qubits_tuple}")
                # print(qubits_tuple in actual_subregisters)
                if qubits_tuple in actual_subregisters:
                    predicted_instruction_subregisters.pop(
                        predicted_instruction_subregisters.index(qubits_tuple)
                    )

            # by popping from the original list, the copies of exact tuples that have exact same contents, we
            # make sure that every gate has appeared exact number of times that we want sub-circuit to be made from

            # If any mismatch will happen, in this test, we either create too much gates, too many Qubits are used, or
            # too little gates are created. Either way we will notice by the contents of the list checked, or the test
            # will result in an error prior to the assert

            self.assertEqual(predicted_instruction_subregisters, [])

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
    def test_chains_gate_optimizer(self):
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

        for this test, a different method of graph creation is used
        """

    def test_compare_methods(self):
        """
        compare both optimizations to check if they yield the same answers (counts may vary)

        compared methods will only be checked in this test for graphs with less or equal to 7 nodes.
        This is due to RAM requirements of "gate" optimizer using a lot more qbits thus matrix is very large.
        """
        # create a series of randomized graphs, from complexity of 4 up to 7 nodes, in series.
        graph_seed = randint(0, 100000)
        seed(graph_seed)
        graphs = [(nodes, self.graph_constructor(nodes)) for nodes in range(4, 8)]
        shots = 10000
        seed_simulator = randint(0, 100000)

        for nodes, (edge_list, _) in graphs:
            gate_optimized_solver = Graph2Cut(
                nodes=nodes, edge_list=edge_list, cuts_number=len(edge_list), optimization='gates')
            qbit_optimized_solver = Graph2Cut(
                nodes=nodes, edge_list=edge_list, cuts_number=len(edge_list), optimization='qbits')
            gate_optimized_solver.solve(shots=shots, seed_simulator=seed_simulator)
            qbit_optimized_solver.solve(shots=shots, seed_simulator=seed_simulator)

            if gate_optimized_solver.possible_answers is None or qbit_optimized_solver.possible_answers is None:
                assert (gate_optimized_solver.possible_answers is None
                        and qbit_optimized_solver.possible_answers is None), \
                    f"both possible answer variables should be None, graph seed: {graph_seed}, " \
                    f"exp. seed {seed_simulator}"
                continue

            q_answers = [a[0] for a in gate_optimized_solver.possible_answers]
            g_answers = [a[0] for a in qbit_optimized_solver.possible_answers]
            assert q_answers == g_answers, f"answers do not match for experiment seed {seed_simulator} " \
                                           f"in compare methods, graph seed: {graph_seed}"

    def test_errors(self):
        """
        test program for running correctness
        """
        rand_chars = 'abcdefghijAFWOPRVZ120987'
        unsupported_operands = ["<", ">", ">=", "<="]
        nodes = 3
        edges = [(0, 1), (1, 2)]
        # (nodes, edges, )
        self.assertRaises(NotImplementedError, Graph2Cut, *[nodes, edges],
                          optimization=''.join([choice(rand_chars) for _ in range(11)]))
        for condition in unsupported_operands:
            self.assertRaises(NotImplementedError, Graph2Cut, *[nodes, edges],
                              condition=condition)

        cut = Graph2Cut(nodes, edges)
        self.assertRaises(RuntimeError, cut.solution_analysis)  # no counts because no experiment was run
        self.assertRaises(RuntimeError, cut.solve, diffusion_iterations=randint(-12580120, -1))
        self.assertRaises(RuntimeError, cut.solve, diffusion_iterations=0)
        self.assertRaises(RuntimeError, cut.construct_circuit_q, diffusion_iterations=0)
        self.assertRaises(RuntimeError, cut.construct_circuit_q, diffusion_iterations=randint(-12580120, -1))
        self.assertRaises(RuntimeError, cut.construct_circuit_g, diffusion_iterations=0)
        self.assertRaises(RuntimeError, cut.construct_circuit_g, diffusion_iterations=randint(-12580120, -1))


if __name__ == '__main__':
    unittest.main()
