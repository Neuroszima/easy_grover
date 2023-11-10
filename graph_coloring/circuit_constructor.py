from typing import Optional

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit


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
        self.results_register: Optional[ClassicalRegister] = None

    def minimal_adder_size(self):
        i = 1
        while len(self.edge_list) >= 2**i:
            i += 1
            if i > 10:
                break
        return i

    def allocate_qbits(self):
        self.node_qbit_register = QuantumRegister(self.graph_nodes)
        self.edge_qbit_register = QuantumRegister(len(self.edge_list))
        self.quantum_adder_register = QuantumRegister(self.minimal_adder_size())
        self.results_register = QuantumRegister(self.graph_nodes)

