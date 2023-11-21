import time
from itertools import pairwise
from typing import Optional
from math import floor, log2
from pprint import pprint

import qiskit.circuit.library
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_aer.backends import AerSimulator
from qiskit.circuit import ControlledGate
from qiskit.circuit.library.standard_gates import XGate
from matplotlib import pyplot as plt
from qiskit.circuit.quantumregister import Qubit


class Graph2Cut:

    def __init__(self, nodes: int, edge_list: list[tuple[int, ...] | list[int, ...]], cuts_number: int = None,
                 condition: str = None, optimization: str = None):
        """
        Perform a graph splitting, based on the graph coloring problem. Only 2 colors are supported by this solver

        For graph splitting, most essential information comes from nodes and edge list, followed
        by the number of cuts to be performed

        "optimization" parameter here makes a tradeoff between number of qbits solution uses and gates.
        "qbits" entry gets rid of entire "edge flagging" part of quantum algorithm, making this up with the
        use of 1 ancilla to store the result temporarily.

        :param nodes: total number of graph members
        :param edge_list: list of 2-member tuples/iterables, that represents graph structure
        :param cuts_number: len(edge_list) is the default
        :param condition: "=" is default
        :param optimization: "gates" or "qbits" are possible modifiers ("gates" is default)
        """
        self.graph_nodes = nodes
        self.edge_list = edge_list
        self.cuts_number = len(self.edge_list) if cuts_number is None else cuts_number
        self.condition = "=" if condition is None else condition
        self.optimization = "gates" if optimization is None else optimization
        self.circuit: Optional[QuantumCircuit] = None
        self.node_qbit_register: Optional[QuantumRegister] = None
        self.edge_qbit_register: Optional[QuantumRegister] = None
        self.quantum_adder_register: Optional[QuantumRegister] = None
        self.ancilla_qbit_register: Optional[QuantumRegister] = None
        self.results_register: Optional[ClassicalRegister] = None
        self.controlled_gate_dict = dict()

        self._allocate_qbits()
        # Create gate dictionary of elements that are required for addition purposes.
        # The most complex (n-C)X has to satisfy the qbit index of the last qbit expressing
        # the most significant power-of-2 bit in classical meaning
        for index, _ in enumerate(self.quantum_adder_register):
            self.controlled_gate_dict[index+1] = XGate().control(index+1)

    def minimal_adder_size(self):
        return floor(log2(len(self.edge_list)))+1

    def _allocate_qbits(self):
        """
        creates registers for future purposes
        """
        self.node_qbit_register = QuantumRegister(self.graph_nodes, name="nodes")
        self.quantum_adder_register = QuantumRegister(self.minimal_adder_size(), name="adder")
        self.results_register = ClassicalRegister(self.graph_nodes, name="measure")
        if self.optimization == "gates":
            self.edge_qbit_register = QuantumRegister(len(self.edge_list), name="edges")
            self.ancilla_qbit_register = QuantumRegister(1, name="ancilla")
        elif self.optimization == 'qbits':
            self.ancilla_qbit_register = QuantumRegister(2, name="ancilla")

    def _edge_flagging(self, surrounding_barriers=True) -> QuantumCircuit:
        """
        generates a circuit part, that is responsible for flagging nodes that are connected
        those will be flagged by converting qbit into state "|1>" when a color mismatch event happen
        (one of qbits will flip the node qbit state with CX into |1>,
        while the other, being |0> will not reverse it into |0> again)
        """
        edge_flagging_circuit_ = QuantumCircuit(self.node_qbit_register, self.edge_qbit_register)

        if surrounding_barriers:
            edge_flagging_circuit_.barrier()

        for edge, target_qbit in zip(self.edge_list, self.edge_qbit_register):
            edge_flagging_circuit_.cx(self.node_qbit_register[edge[0]], target_qbit)
            edge_flagging_circuit_.cx(self.node_qbit_register[edge[1]], target_qbit)

        if surrounding_barriers:
            edge_flagging_circuit_.barrier()

        return edge_flagging_circuit_

    def _grover_diffusion(self):
        """
        flags all the states that present themselves to solve the equation, inverting their state for "-"
        """

        grover_diffusion_circuit = QuantumCircuit(self.node_qbit_register)
        x_gate_controlled = XGate().control(num_ctrl_qubits=len(self.node_qbit_register)-1)
        # z_gate_controlled = qiskit.circuit.library.ZGate().control(num_ctrl_qubits=len(self.node_qbit_register)-1)

        grover_diffusion_circuit.h(self.node_qbit_register)
        grover_diffusion_circuit.x(self.node_qbit_register)

        # I had troubles with qiskit recognizing n-(c)ZGate, this construct seems to work as an equivalent
        grover_diffusion_circuit.h(self.node_qbit_register[-1])
        grover_diffusion_circuit.append(x_gate_controlled, [*self.node_qbit_register])
        grover_diffusion_circuit.h(self.node_qbit_register[-1])
        # grover_diffusion_circuit.append(z_gate_controlled, self.node_qbit_register)

        grover_diffusion_circuit.x(self.node_qbit_register)
        grover_diffusion_circuit.h(self.node_qbit_register)

        return grover_diffusion_circuit

    def _adder(self):
        """
        prepares a circuit that counts edges based on the coloring scheme
        lay out the circuit based on the total number of 1 bit additions required for the algorithm to complete
        """

        grover_adder_circuit = QuantumCircuit(self.edge_qbit_register, self.quantum_adder_register)

        all_gates = []
        for qbit_index, qbit in enumerate(self.edge_qbit_register):

            # there needs to be a gate pyramid created for each qbit, with control complexity
            # that match floor(log2) of exact qbit in a series to be added.

            temp = floor(int(log2(qbit_index+1))) + 1
            # print(f"{temp=} now")
            gates_for_qbit = []
            while temp > 0:
                gates_for_qbit.append([qbit, self.controlled_gate_dict[temp]])
                temp -= 1
            all_gates.extend(gates_for_qbit)

        # pprint(all_gates)

        for instruction_pack in all_gates:
            # here we add gates with control and make a part of the circuit
            # based on target that was passed in previous step
            # we might merge these loops in future release

            qbit: Qubit = instruction_pack[0]
            gate: ControlledGate = instruction_pack[1]

            control_adder_qbits_count = target_adder_bit = gate.num_qubits - 2
            if control_adder_qbits_count == 0:
                control_adder_qbits = []
            else:
                control_adder_qbits = [self.quantum_adder_register[i] for i in range(control_adder_qbits_count)]
            grover_adder_circuit.append(
                gate, [qbit, *control_adder_qbits, self.quantum_adder_register[target_adder_bit]]
            )

        return grover_adder_circuit

    def _condition_checking(self):
        """
        prepare a part of the circuit that makes up a condition
        for a phase flip for solutions space (into "-")
        """
        if not self.condition == "=":
            raise NotImplementedError("other types of comparisons other than '=' are not supported yet")

        checker_circuit = QuantumCircuit(self.quantum_adder_register, self.ancilla_qbit_register)
        checker_circuit.barrier()
        # for equality, we flip the bits that should be "0" in the bitwise representation
        # of given integer, then we "AND" entire adder register together -> making sure all the bits are proper state

        string_of_num = f"{bin(self.cuts_number)[2:]}".zfill(len(self.quantum_adder_register))
        for index, digit in enumerate(reversed(string_of_num)):
            # apply X wherever there is "0"
            if digit == "0":
                checker_circuit.x(self.quantum_adder_register[index])

        MCXGate = XGate().control(len(self.quantum_adder_register))
        checker_circuit.append(MCXGate, [*self.quantum_adder_register, self.ancilla_qbit_register[-1]])
        return checker_circuit

    def _add_single_edge(self, edge_index: int, edge_0_qbit, edge_1_qbit, ancilla_qbit):
        """prepare a small sub-circuit that adds the edge check result to the accumulator"""

        # create a single adder pyramid
        temp = floor(int(log2(edge_index + 1))) + 1
        # print(f"{temp=} now")
        gates_for_qbit: list[ControlledGate] = []
        # compared to forming an adder circuit openly, here we always use the same ancilla qbit as target for gate
        while temp > 0:
            gates_for_qbit.append(self.controlled_gate_dict[temp])
            temp -= 1

        # take adder qbits that express highest power-of-2 representation of the edge check result
        # being added, as if all of them were '1', -1 because one of the participants is ancilla qbit
        edge_addition_circuit = QuantumCircuit([
            edge_0_qbit, edge_1_qbit, ancilla_qbit,
            *self.quantum_adder_register[:gates_for_qbit[0].num_qubits-1]
        ])
        edge_addition_circuit.cx(edge_0_qbit, ancilla_qbit)
        edge_addition_circuit.cx(edge_1_qbit, ancilla_qbit)

        for controlled_gate in gates_for_qbit:

            control_adder_qbits_count = target_adder_bit = controlled_gate.num_qubits - 2
            if control_adder_qbits_count == 0:
                control_adder_qbits = []
            else:
                control_adder_qbits = [self.quantum_adder_register[i] for i in range(control_adder_qbits_count)]

            edge_addition_circuit.append(
                controlled_gate, [ancilla_qbit, *control_adder_qbits, self.quantum_adder_register[target_adder_bit]]
            )

        edge_addition_circuit.cx(edge_1_qbit, ancilla_qbit)
        edge_addition_circuit.cx(edge_0_qbit, ancilla_qbit)

        # print(f"circuit for compressed addition, {edge_index=}")
        # edge_addition_circuit.draw(output='mpl', )
        # plt.show()
        # print(edge_addition_circuit)
        return edge_addition_circuit

    def assemble_subcircuits(self):
        """prepare all the useful sub-circuits to be merged into main one later"""
        diffusion_circuit = self._grover_diffusion()
        condition_check_circuit = self._condition_checking()
        if self.optimization == "gates":
            edge_checking_circuit = self._edge_flagging()
            adder_circuit = self._adder()

            pack = edge_checking_circuit, adder_circuit, diffusion_circuit, condition_check_circuit
            # for c in pack:
            #     print(pack)
            return pack
        elif self.optimization == "qbits":
            adder_subcircuits = []
            for edge_index, edge in enumerate(self.edge_list):
                subcircuit = self._add_single_edge(
                    edge_index=edge_index,
                    edge_0_qbit=self.node_qbit_register[edge[0]],
                    edge_1_qbit=self.node_qbit_register[edge[1]],
                    ancilla_qbit=self.ancilla_qbit_register[0],
                    # adder_qbits=[*self.quantum_adder_register[:]]
                )
                adder_subcircuits.append(subcircuit)
            return adder_subcircuits, diffusion_circuit, condition_check_circuit
        raise NotImplementedError(f"optimization method {self.optimization} not implemented as valid optimization")

    def construct_circuit(self, diffusion_iterations=3):
        """
        based on the iterations passed into this function, create a sequence of
        gates to be applied in the quantum circuit

        the basic construction chain goes like this:

        do following param algo_iteration times:
            - check edges (if colors do differ indeed)
            - add edges that satisfy conditions
            - perform a check based on the added edges
            - flip the phase
            - reverse previous operations to propagate flipped phase
            - apply grovers diffusion

        finally - measure outcomes

        :param diffusion_iterations: the number of full steps to achieve probability of
            finding solutions satisfactory
        """
        self.circuit = QuantumCircuit(
            self.node_qbit_register, self.edge_qbit_register,
            self.quantum_adder_register, self.ancilla_qbit_register,
            self.results_register
        )
        self.circuit.h(self.node_qbit_register)

        edge_checker, adder, diffusion, condition_checker = self.assemble_subcircuits()
        # print("in circuits check")
        # print(f"edge_checker\n{edge_checker}")
        # print(f"adder\n{adder}")
        # print(f"diffusion\n{diffusion}")
        # print(f"condition_checker\n{condition_checker}")

        for _ in range(diffusion_iterations):
            # oracle step
            self.circuit.compose(edge_checker, inplace=True)
            self.circuit.compose(adder, [*self.edge_qbit_register, *self.quantum_adder_register], inplace=True)
            self.circuit.compose(
                condition_checker, [*self.quantum_adder_register, self.ancilla_qbit_register[0]], inplace=True)
            self.circuit.z(self.ancilla_qbit_register[0])
            self.circuit.compose(
                condition_checker.reverse_ops(),
                [*self.quantum_adder_register, self.ancilla_qbit_register[0]],
                inplace=True
            )
            self.circuit.compose(
                adder.reverse_ops(), [*self.edge_qbit_register, *self.quantum_adder_register], inplace=True)
            self.circuit.compose(
                edge_checker.reverse_ops(), [*self.node_qbit_register, *self.edge_qbit_register], inplace=True)

            # diffusion step
            self.circuit.compose(diffusion, [*self.node_qbit_register], inplace=True)

        self.circuit.barrier()
        self.circuit.measure(qubit=self.node_qbit_register, cbit=self.results_register)

    def schedule_job_locally(self, shots=1000):
        """
        run circuit measurements locally on your PC with standard settings

        default simulator to use is 'qasm' that provides only counts and measurements, but any can be used

        :returns: job results
        """
        # print(self.circuit)
        job = AerSimulator().run(self.circuit, shots=shots)
        counts = job.result().get_counts(self.circuit)
        return counts


if __name__ == '__main__':
    nodes = 3
    edges = [(0, 1), (1, 2)]
    cut = Graph2Cut(nodes, edges, len(edges))
    circuit_book = cut.assemble_subcircuits()
    cut.construct_circuit(diffusion_iterations=1)
    counts = cut.schedule_job_locally()
    print(sorted([(ans, counts[ans]) for ans in counts], key=lambda x: x[1])[::-1])

    # nodes2 = 11
    # edges2 = [*pairwise(i for i in range(nodes2))]
    # cut2 = Graph2Cut(nodes2, edges2, len(edges2))
    # circuit_book2 = cut2.assemble_subcircuits()
    # cut2.construct_circuit(diffustion_iterations=3)
    # st = time.time()
    # counts2 = cut2.schedule_job_locally(shots=30000)
    # ed = time.time()
    # print("time", ed - st)
    # print(sorted([(ans, counts2[ans]) for ans in counts2], key=lambda x: x[1])[::-1])
