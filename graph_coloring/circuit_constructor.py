import time
from itertools import pairwise
from typing import Optional
from math import floor, log2
from pprint import pprint

import qiskit.circuit.library
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
from qiskit.circuit import ControlledGate
from qiskit.circuit.library.standard_gates import XGate
from matplotlib import pyplot as plt
from qiskit.circuit.quantumregister import Qubit


class Graph2Cut:

    def __init__(self, nodes: int, edge_list: list[tuple[int, ...] | list[int, ...]], cuts_number: int,
                 condition: str = None):
        self.graph_nodes = nodes
        self.edge_list = edge_list
        self.cuts_number = cuts_number
        self.condition = "=" if condition is None else condition
        self.circuit: Optional[QuantumCircuit] = None
        self.node_qbit_register: Optional[QuantumRegister] = None
        self.edge_qbit_register: Optional[QuantumRegister] = None
        self.quantum_adder_register: Optional[QuantumRegister] = None
        self.ancilla_qbit_register: Optional[QuantumRegister] = None
        self.results_register: Optional[ClassicalRegister] = None

    def minimal_adder_size(self):
        return floor(log2(len(self.edge_list)))+1

    def _allocate_qbits(self):
        """
        creates registers for future purposes
        """
        self.node_qbit_register = QuantumRegister(self.graph_nodes, name="nodes")
        self.edge_qbit_register = QuantumRegister(len(self.edge_list), name="edges")
        self.quantum_adder_register = QuantumRegister(self.minimal_adder_size(), name="adder")
        self.results_register = ClassicalRegister(self.graph_nodes, name="measure")
        self.ancilla_qbit_register = QuantumRegister(1, name="ancilla")

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
        z_gate_controlled = qiskit.circuit.library.ZGate().control(num_ctrl_qubits=len(self.node_qbit_register)-1)

        grover_diffusion_circuit.h(self.node_qbit_register)
        grover_diffusion_circuit.x(self.node_qbit_register)
        grover_diffusion_circuit.append(z_gate_controlled, self.node_qbit_register)
        grover_diffusion_circuit.x(self.node_qbit_register)
        grover_diffusion_circuit.h(self.node_qbit_register)

        return grover_diffusion_circuit

    def _adder(self):
        """
        prepares a circuit that counts edges based on the coloring scheme
        lay out the circuit based on the total number of 1 bit additions required for the algorithm to complete
        """

        grover_adder_circuit = QuantumCircuit(self.edge_qbit_register, self.quantum_adder_register)
        controled_gate_dict = dict()

        # Create gate dictionary of elements that are required for addition purposes.
        # The most complex (n-C)X has to satisfy the qbit index of the last qbit being in the order of.
        for index, _ in enumerate(self.quantum_adder_register):
            controled_gate_dict[index+1] = XGate().control(index+1)

        # print(controled_gate_dict)

        all_gates = []
        for qbit_index, qbit in enumerate(self.edge_qbit_register):

            # there needs to be a gate pyramid created for each qbit, with control complexity
            # that match floor(log2) of exact qbit in a series to be added.

            temp = floor(int(log2(qbit_index+1))) + 1
            # print(f"{temp=} now")
            gates_for_qbit = []
            while temp > 0:
                gates_for_qbit.append([qbit, controled_gate_dict[temp]])
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
        print(string_of_num)
        for index, digit in enumerate(reversed(string_of_num)):
            # apply X whenever there is "0"
            if digit == "0":
                checker_circuit.x(self.quantum_adder_register[index])

        MCXGate = XGate().control(len(self.quantum_adder_register))
        checker_circuit.append(MCXGate, [*self.quantum_adder_register, self.ancilla_qbit_register[0]])
        return checker_circuit

    def assemble_subcircuits(self):
        """prepare all the useful sub-circuits to be merged into main one later"""
        self._allocate_qbits()
        edge_checking_circuit = self._edge_flagging()
        adder_circuit = self._adder()
        diffusion_circuit = self._grover_diffusion()
        condition_check_circuit = self._condition_checking()

        pack = edge_checking_circuit, adder_circuit, diffusion_circuit, condition_check_circuit
        # for c in pack:
        #     print(pack)
        return pack

    def construct_circuit(self, diffustion_iterations=3):
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
        :param diffustion_iterations: the number of full steps to achieve probability of
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

        for _ in range(diffustion_iterations):
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

    def schedule_job_locally(self, backend="qasm_simulator", shots=1000):
        """
        run circuit measurements locally on your PC with standard settings

        default simulator to use is 'qasm' that provides only counts and measurements, but any can be used
        :param backend: simulator backend to use in job scheduler
        :returns: job results
        """

        job = execute(self.circuit, Aer.get_backend(backend), shots=shots)
        counts = job.result().get_counts(self.circuit)
        return counts


if __name__ == '__main__':
    nodes = 3
    edges = [(0, 1), (1, 2)]
    nodes2 = 13
    edges2 = [*pairwise(i for i in range(nodes2))]
    cut = Graph2Cut(nodes, edges, len(edges), "=")
    cut2 = Graph2Cut(nodes2, edges2, len(edges2), "=")
    circuit_book = cut.assemble_subcircuits()
    circuit_book2 = cut2.assemble_subcircuits()
    cut.construct_circuit(diffustion_iterations=1)
    cut2.construct_circuit(diffustion_iterations=3)
    counts = cut.schedule_job_locally()
    st = time.time()
    counts2 = cut2.schedule_job_locally(shots=30000)
    ed = time.time()
    print("time", ed - st)
    print(sorted([(ans, counts[ans]) for ans in counts], key=lambda x: x[1])[::-1])
    print(sorted([(ans, counts2[ans]) for ans in counts2], key=lambda x: x[1])[::-1])
