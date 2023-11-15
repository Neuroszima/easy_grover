from typing import Optional
from math import floor, log2
from pprint import pprint

import qiskit.circuit.library
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import ControlledGate
from qiskit.circuit.library.standard_gates import XGate
from matplotlib import pyplot as plt
from qiskit.circuit.quantumregister import Qubit


class Graph2Cut:

    def __init__(self, nodes: int, edge_list: list[tuple[int, ...] | list[int, ...]], cuts_number: int, condition: str):
        self.graph_nodes = nodes
        self.edge_list = edge_list
        self.cuts_number = cuts_number
        self.condition = condition
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
        self.node_qbit_register = QuantumRegister(self.graph_nodes)
        self.edge_qbit_register = QuantumRegister(len(self.edge_list))
        self.quantum_adder_register = QuantumRegister(self.minimal_adder_size())
        self.results_register = QuantumRegister(self.graph_nodes)
        self.ancilla_qbit_register = QuantumRegister(1)

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

    def assemble_circuits(self):
        self._allocate_qbits()
        edge_checking_circuit = self._edge_flagging()
        adder_circuit = self._adder()
        diffusion_circuit = self._grover_diffusion()
        return edge_checking_circuit, adder_circuit, diffusion_circuit
        # c = self._adder()
        # circuit = cut._grover_diffusion()
        # obj = c.draw(output="mpl")
        # print(obj)
        # plt.show()


if __name__ == '__main__':
    nodes = 3
    edges = [(0, 1), (1, 2)]
    nodes2 = 7
    edges2 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    cut = Graph2Cut(nodes, edges, len(edges), "=")
    cut2 = Graph2Cut(nodes2, edges2, len(edges2), "=")
    cut._allocate_qbits()
    cut2._allocate_qbits()
    # circuit = cut._grover_diffusion()
    # obj = circuit.draw(output="mpl")
    # print(obj)
    # plt.show()
    # circuit2 = cut2._grover_diffusion()
    # obj2 = circuit2.draw(output="mpl")
    # print(obj2)
    # plt.show()
    # cut.test_circuits()
    # cut2.test_circuits()
