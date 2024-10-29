import warnings
from collections import OrderedDict
from pprint import pprint
from typing import Optional
from math import floor, log2

import numpy as np
from matplotlib import pyplot as plt
# import qiskit.circuit.library
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_aer.backends import AerSimulator
from qiskit.circuit import ControlledGate
from qiskit.circuit.library.standard_gates import XGate
from qiskit.circuit.quantumregister import Qubit

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroll3qOrMore

from numpy import ndarray


class Graph2Cut:

    ALLOWED_OPTIMIZERS = ["gates", "qbits"]
    ALLOWED_CONDITIONS = ["="]

    def __init__(self, nodes: int, edges: list[tuple[int, int] | list[int, int]] | ndarray,
                 cuts_number: int = None, condition: str = None, optimization: str = None,
                 allow_experimental_runs=False, hardware_simulation_memory_limit: str | None = None):
        """
        Perform a graph splitting, based on the graph coloring problem. Only 2 colors are supported by this solver

        For graph splitting, most essential information comes from nodes and edge list, followed
        by the number of cuts to be performed

        "optimization" parameter here makes a tradeoff between number of qbits solution uses and gates.
        "qbits" entry gets rid of entire "edge flagging" part of quantum algorithm, making this up with the
        use of 1 ancilla to store the result temporarily.

        Note: determine the notation order in which answers are served and understand its meaning

        :param nodes: total number of graph members
        :param edges: list of 2-member tuples/iterables, that represents graph structure
        :param cuts_number: len(edge_list) is the default
        :param condition: "=" is default
        :param optimization: "gates" or "qbits" are possible modifiers ("gates" is default)
        :param allow_experimental_runs: allows modes and conditions that could result in program errors
        :param hardware_simulation_memory_limit: this param sets RAM usage when interacting with Aer backend
            ("8GB" default, set lower if you have lower end hardware)
        """
        self.allow_experimental_runs = allow_experimental_runs
        self.graph_nodes = nodes
        if isinstance(edges, ndarray):
            self.edge_list = self.translate_matrix_representation(edges)
        elif isinstance(edges, (list, tuple)):
            self.edge_list = edges
        else:
            raise TypeError(f"edges has to be list/tuple of pairs of nodes, or 2D matrix, not {type(edges)}")
        self.cuts_number = len(self.edge_list) if cuts_number is None else cuts_number
        self.max_cut = len(self.edge_list)

        if allow_experimental_runs:
            warnings.warn(
                "if used with unsupported modes or conditions, this can yield errors use at own risk"
            )

        if allow_experimental_runs:
            self.condition = condition
            self.min_range, self.max_range = self._checked_ranges(cuts_number, condition)
        elif condition in [*self.ALLOWED_CONDITIONS, None]:
            self.condition = "=" if condition is None else condition
            self.min_range, self.max_range = self._checked_ranges(cuts_number, self.condition)
        else:
            raise NotImplementedError("other types of comparisons other than '=' are not supported yet")

        if allow_experimental_runs:
            self.optimization = optimization
        elif optimization in [*self.ALLOWED_OPTIMIZERS, None]:
            self.optimization = "gates" if optimization is None else optimization
        else:
            raise NotImplementedError(f'optimization: {optimization} not recognized as valid optimization')

        self.hardware_mem_limit = "8GB" if hardware_simulation_memory_limit is None \
            else hardware_simulation_memory_limit

        self.circuit: Optional[QuantumCircuit] = None
        self.node_qbit_register: Optional[QuantumRegister] = None
        self.edge_qbit_register: Optional[QuantumRegister] = None
        self.quantum_adder_register: Optional[QuantumRegister] = None
        self.ancilla_qbit_register: Optional[QuantumRegister] = None
        self.results_register: Optional[ClassicalRegister] = None
        self.controlled_gate_dict = dict()
        self.answer_percentage: Optional[float] = None
        self.current_job_shots: Optional[int] = None
        self.counts: Optional[dict] = None

        self._allocate_qbits()
        # Create gate dictionary of elements that are required for addition purposes.
        # The most complex (n-C)X has to satisfy the qbit index of the last qbit expressing
        # the most significant power-of-2 bit in classical meaning
        # THIS MAPPING STARTS FROM "1" AS STARTING INDEX
        for index, _ in enumerate(self.quantum_adder_register):
            self.controlled_gate_dict[index+1] = XGate().control(index+1)

        # uninitialized state
        self._gates_total: Optional[int, OrderedDict] = 0
        self.possible_answers: Optional[list[tuple[str, int]]] = None
        self.diffusion_steps = 0
        self._qbits_total = 0

    @staticmethod
    def translate_matrix_representation(matrix: ndarray) -> list:
        """
        method for transforming graph representation from np.ndarray 2D matrix into list of edges
        method always takes upper right part of the matrix into consideration (symmetry assumed), with respect
        to diagonal

        when non-symmetric matrix is passed, the lower left part is ignored and does not throw an error
        self-references in the graph (diagonals) are also ignored
        """
        # direction at which our graphs edges are oriented does not matter.
        # we always take symmetrical approach, so only one part of matrix is used
        if len(matrix.shape) != 2:
            raise ValueError("matrix form of graph should be of 2D shape")
        i = 0
        conn_list = []
        while i < matrix.shape[0]:
            j = i + 1
            conn_list.extend([(i, x) for x in range(j, matrix.shape[0]) if matrix[i][x]])
            i += 1
        return conn_list

    def size(self, as_dict=True):
        """
        get basic information about solution cost
        uses circuit translator to unroll >3q-controlled gates, to gauge the performance on actual QC

        :param as_dict: forces to return value, otherwise prints it out in beautified form
        """
        if isinstance(self.circuit, QuantumCircuit):
            pass_manager = PassManager(Unroll3qOrMore(basis_gates=['cx', 'u3']))
            new_circuit = pass_manager.run(self.circuit)
            self._gates_total: OrderedDict = new_circuit.count_ops()

        regiters = [
            self.node_qbit_register, self.edge_qbit_register,
            self.quantum_adder_register, self.ancilla_qbit_register
        ]
        self._qbits_total = sum([len(register) if register else 0 for register in regiters])
        d = {
            "c_bits": self.results_register.size,
            "q_bits": self._qbits_total,
            "base_instruction_count": self._gates_total  # unable to calculate
        }
        if as_dict:
            return d
        else:
            for key in d:
                if key == "base_instruction_count":
                    print("%s: {" % key)
                    for k_ in d[key]:
                        print(f"    {k_}: {d[key][k_]},")
                    print("}")
                else:
                    print(f"{key}: {d[key]},")

    def _simulation_memory_usage_limit(self) -> int:
        """calculates how many MB of memory to use when specified"""

    def _minimal_adder_size(self):
        return floor(log2(len(self.edge_list)))+1

    def _checked_ranges(self, cuts_number, condition):
        """set brackets for """
        if condition == ">":
            min_range = cuts_number + 1
            max_range = len(self.edge_list)
        elif condition == ">=":
            min_range = cuts_number
            max_range = len(self.edge_list)
        elif condition == "<":
            max_range = cuts_number - 1
            min_range = 0
        elif condition == "<=":
            max_range = cuts_number
            min_range = 0
        elif condition == "=":
            max_range = min_range = cuts_number
        else:
            raise ValueError(f"improper condition: {condition}")
        return min_range, max_range

    def _allocate_qbits(self):
        """
        creates registers for future purposes
        """
        self.node_qbit_register = QuantumRegister(self.graph_nodes, name="nodes")
        self.quantum_adder_register = QuantumRegister(self._minimal_adder_size(), name="adder")
        self.results_register = ClassicalRegister(self.graph_nodes, name="measure")
        if self.optimization == "gates":
            self.edge_qbit_register = QuantumRegister(len(self.edge_list), name="edges")
            self.ancilla_qbit_register = QuantumRegister(1, name="ancilla")
        elif self.optimization == 'qbits':
            self.ancilla_qbit_register = QuantumRegister(2, name="ancilla")

    def _edge_flagging(self, surrounding_barriers=True) -> QuantumCircuit:
        """
        Generates a circuit part, that is responsible for flagging nodes that are connected.

        those will be flagged by converting qbit into state "|1>" when a color mismatch event happen
        (one of qubits will flip the node qbit state with CX into |1>, while the other, being |0>, will
        not reverse it into |0> again). This is a part of procedure known as "quantum counting"

        This very method only triggers when selected optimization is "gates"
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
        n-controlled z-gate in this stage is not supported by qiskit, so it was swapped into x-gate based
        diffusion mechanism
        """

        grover_diffusion_circuit = QuantumCircuit(self.node_qbit_register)
        x_gate_controlled = XGate().control(num_ctrl_qubits=len(self.node_qbit_register)-1)

        grover_diffusion_circuit.h(self.node_qbit_register)
        grover_diffusion_circuit.x(self.node_qbit_register)

        # I had troubles with qiskit recognizing n-(c)ZGate, this construct seems to work as an equivalent
        grover_diffusion_circuit.h(self.node_qbit_register[-1])
        grover_diffusion_circuit.append(x_gate_controlled, [*self.node_qbit_register])
        grover_diffusion_circuit.h(self.node_qbit_register[-1])

        grover_diffusion_circuit.x(self.node_qbit_register)
        grover_diffusion_circuit.h(self.node_qbit_register)

        return grover_diffusion_circuit

    def _adder(self):
        """
        prepares a circuit that counts edges based on the coloring scheme
        lays out the circuit based on the total number of 1 bit additions required for the algorithm to complete
        """

        grover_adder_circuit = QuantumCircuit(self.edge_qbit_register, self.quantum_adder_register)

        all_gates = []
        for qbit_index, qbit in enumerate(self.edge_qbit_register):

            # there needs to be a gate pyramid created for each qbit, with control complexity
            # that match floor(log2) of exact qbit in a series to be added.

            temp = floor(int(log2(qbit_index+1))) + 1
            gates_for_qbit = []
            while temp > 0:
                gates_for_qbit.append([qbit, self.controlled_gate_dict[temp]])
                temp -= 1
            all_gates.extend(gates_for_qbit)

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

        checker_circuit = QuantumCircuit(self.quantum_adder_register, self.ancilla_qbit_register)
        checker_circuit.barrier()
        # for equality, we flip the bits that should be "0" in the bitwise representation
        # of given integer, then we "AND" entire adder register together -> making sure all the bits are proper state

        bitstring = f"{bin(self.cuts_number)[2:]}".zfill(len(self.quantum_adder_register))
        for index, digit in enumerate(reversed(bitstring)):
            # apply X wherever there is "0"
            if digit == "0":
                checker_circuit.x(self.quantum_adder_register[index])

        MCXGate = XGate().control(len(self.quantum_adder_register))
        checker_circuit.append(MCXGate, [*self.quantum_adder_register, self.ancilla_qbit_register[-1]])
        return checker_circuit

    def _form_condition_check_from_bitstring(self, bitstring: str, checker_circuit: QuantumCircuit):
        """
        take a string of bits as an input; determine which qbits should be negated, and at the end, which
        qbits should be checked against.

        based on string contents:
        1 -> check this qbit as is (include into control bits)
        0 -> check negation of this qbit (include into control bits)
        X -> do not check this qbit (i.e. this qbit can have any value, does not matter)
        """
        # from matplotlib import pyplot as plt
        print(bitstring)
        negate_array = []
        control_qbits = []
        for index, bit in enumerate(reversed(bitstring)):
            if bit != "X":
                control_qbits.append(self.quantum_adder_register[index])
            if bit == "0":
                negate_array.append(index)

        print(f"stripped {bitstring.strip('X')}")
        controls_count = len(bitstring.strip("X"))  # +1
        MCXGate = XGate().control(controls_count)
        # below -> the first control qbit that we should consider in 0101XXXXX case
        # we should take all the controls after this one
        try:
            earlies_x_pos = bitstring.index("X")
        except ValueError:
            earlies_x_pos = len(bitstring)
        control_state_begin = len(bitstring) - earlies_x_pos
        print(controls_count)
        print(control_state_begin)

        # mark all the bits, use MCXGate as multi-check preparation, mark all the "correct states" with Z and reverse
        print([*self.quantum_adder_register[control_state_begin:], self.ancilla_qbit_register[-1]])
        for neg in negate_array:
            checker_circuit.append(XGate(), [self.quantum_adder_register[neg]])
        checker_circuit.append(MCXGate, [
            *self.quantum_adder_register[control_state_begin:], self.ancilla_qbit_register[-1]])
        checker_circuit.z(self.ancilla_qbit_register[-1])
        checker_circuit.append(MCXGate, [
            *self.quantum_adder_register[control_state_begin:], self.ancilla_qbit_register[-1]])
        # mini bit-set reverse for the next check, but still prior to ZGate()
        for neg in negate_array:
            checker_circuit.append(XGate(), [self.quantum_adder_register[neg]])

        # checker_circuit.draw(output='mpl')
        # plt.show()

        return checker_circuit

    def _complex_condition_checking(self):
        """
        prepare a part of the circuit that makes up a condition for a phase flip for solutions space (into "-")

        compared to simple condition checker, this one can handle ranges, as well as greater/lower than cases
        """
        # prepare several control gates based on cumulative coverage of all integers within the checked range

        # base case that i thought in the beginning was to merge integer cases based on single bit difference
        # for example (loose chain of thought):
        #     len(edges) = 14
        #     condition ->  "8>" (greater than 8)
        #     sample integers from the range: 9, 10, 11, 12, 13, 14
        #     bit repr(11) = 0b0000_1011  (i always base it off 8-bit, easier to think about)
        #     bit repr(10) = 0b0000_1010  (i always base it off 8-bit, easier to think about)
        #     example 11 and example 10 can be represented as 0b0000_101X where "X" is "whatever"
        #     this means we can skip the last bit in check, redicing MCXGate count by 1 and control-bit by 1 as well,
        #     at the same time covering both cases.
        #     bit repr(12) = 0b0000_1100
        #     bit repr(13) = 0b0000_1101
        #        (12, 13) -> 0b0000_110X ...merged and so on, then merging "110X" representations into "11XX" and so on

        # here we will use this exact method
        if self.condition == ">":
            r = range(self.cuts_number+1, self.max_cut+1)  # "+1" -> full
        elif self.condition == ">=":
            r = range(self.cuts_number, self.max_cut+1)
        elif self.condition == "<=":
            r = range(0, self.cuts_number+1)
        elif self.condition == "<":
            r = range(0, self.cuts_number)
        else:
            raise RuntimeError(
                f"Can't formulate proper range to build conditional checker with this condition: {self.condition}")

        # since we always check the last bit, we will successively remove neighbouring numbers with single
        # bit of difference, and then leftshift afterward. This marks we found 2 numbers that can be checked with
        # one condition.
        # Do this process for as long as necessary.
        # below, collection of tuples -> (number,
        starting_conditions = [(i, 0) for i in r]
        intermediate_conditions = []
        changed = True
        while changed:
            print(f"{starting_conditions=}")
            changed = False
            while len(starting_conditions) > 1:
                if ((starting_conditions[0][0] ^ starting_conditions[1][0]) == 1 and
                        (starting_conditions[0][1] == starting_conditions[1][1])):
                    # the same shift and single bit difference, due to being 1 away from each other
                    intermediate_conditions.append(
                        (starting_conditions[1][0] >> 1, starting_conditions[1][1] + 1)
                    )
                    starting_conditions.pop(0)
                    starting_conditions.pop(0)
                    changed = True
                else:
                    intermediate_conditions.append(starting_conditions.pop(0))
                if len(starting_conditions) == 1:
                    intermediate_conditions.append(starting_conditions.pop())
            starting_conditions = intermediate_conditions
            intermediate_conditions = []

        # when "gate variety" is prepared, finally prepare a subcircuit
        checker_circuit = QuantumCircuit(self.quantum_adder_register, self.ancilla_qbit_register)
        checker_circuit.barrier()

        for upper_bits, shift in starting_conditions:  # add every condition check with Z and immediate reverse
            gate_definition = f"{bin(upper_bits)[2:]}" + "X"*shift
            print(f"initial gate definition: {gate_definition}")
            gate_definition = gate_definition.zfill(len(self.quantum_adder_register))
            print(f"morphed gate definition: {gate_definition}")
            checker_circuit = self._form_condition_check_from_bitstring(
                bitstring=gate_definition, checker_circuit=checker_circuit)

        checker_circuit.barrier()

        return checker_circuit

    def _add_single_edge(self, edge_index: int, edge_0_qbit, edge_1_qbit, ancilla_qbit):
        """prepare a small sub-circuit that adds the edge check result to the accumulator"""

        # create a single adder pyramid
        temp = floor(log2(edge_index + 1)) + 1
        adder_gates_required: list[ControlledGate] = []
        # compared to forming an adder circuit openly, here we always use the same ancilla qbit as target for gate
        while temp > 0:
            adder_gates_required.append(self.controlled_gate_dict[temp])
            temp -= 1

        # take adder qbits that express highest power-of-2 representation of the edge check result
        # being added, as if all of them were '1', -1 because one of the participants is ancilla qbit
        edge_addition_circuit = QuantumCircuit([
            edge_0_qbit, edge_1_qbit, ancilla_qbit,
            *self.quantum_adder_register[:adder_gates_required[0].num_qubits-1]
        ])
        edge_addition_circuit.cx(edge_0_qbit, ancilla_qbit)
        edge_addition_circuit.cx(edge_1_qbit, ancilla_qbit)

        for controlled_gate in adder_gates_required:

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

        return edge_addition_circuit

    def assemble_subcircuits(self):
        """prepare all the useful sub-circuits to be merged into main one later"""
        diffusion_circuit = self._grover_diffusion()

        if self.condition == "=":
            condition_check_circuit = self._condition_checking()
        else:
            condition_check_circuit = self._complex_condition_checking()

        if self.optimization == "gates":
            edge_checking_circuit = self._edge_flagging()
            adder_circuit = self._adder()

            pack = edge_checking_circuit, adder_circuit, diffusion_circuit, condition_check_circuit
            return pack
        elif self.optimization == "qbits":
            adder_subcircuits_collection = []
            for edge_index, edge in enumerate(self.edge_list):
                subcircuit = self._add_single_edge(
                    edge_index=edge_index,
                    edge_0_qbit=self.node_qbit_register[edge[0]],
                    edge_1_qbit=self.node_qbit_register[edge[1]],
                    ancilla_qbit=self.ancilla_qbit_register[0],
                )
                adder_subcircuits_collection.append([edge_index, edge, subcircuit])
            return adder_subcircuits_collection, diffusion_circuit, condition_check_circuit
        raise NotImplementedError(f"optimization method {self.optimization} not implemented as valid optimization")

    def construct_circuit_g(self, diffusion_iterations=1, use_new_checker=False):
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
        if diffusion_iterations < 1:
            raise RuntimeError('Diffusion iteration step count cannot be lower than 1')
        self.circuit = QuantumCircuit(
            self.node_qbit_register, self.edge_qbit_register,
            self.quantum_adder_register, self.ancilla_qbit_register,
            self.results_register
        )
        self.circuit.h(self.node_qbit_register)

        edge_checker, adder, diffusion, condition_checker = self.assemble_subcircuits()

        for _ in range(diffusion_iterations):
            # oracle step
            self.circuit.compose(edge_checker, inplace=True)
            self.circuit.compose(adder, [*self.edge_qbit_register, *self.quantum_adder_register], inplace=True)

            # major internal change pending, old way left for previous compatibility
            if not use_new_checker:
                self.circuit.compose(
                    condition_checker, [*self.quantum_adder_register, self.ancilla_qbit_register[0]], inplace=True)
                self.circuit.z(self.ancilla_qbit_register[0])
                self.circuit.compose(
                    condition_checker.reverse_ops(),
                    [*self.quantum_adder_register, self.ancilla_qbit_register[0]],
                    inplace=True
                )
                self.circuit.draw(output='mpl')
                plt.show()
            else:
                # this uses checker that is more robust and far more broad than simple one
                self.circuit.compose(
                    condition_checker, [*self.quantum_adder_register, self.ancilla_qbit_register[0]], inplace=True)
            self.circuit.compose(
                adder.reverse_ops(), [*self.edge_qbit_register, *self.quantum_adder_register], inplace=True)
            self.circuit.compose(
                edge_checker.reverse_ops(), [*self.node_qbit_register, *self.edge_qbit_register], inplace=True)

            # diffusion step
            self.circuit.compose(diffusion, [*self.node_qbit_register], inplace=True)

        self.circuit.barrier()
        self.circuit.measure(qubit=self.node_qbit_register, cbit=self.results_register)
        # self.circuit.draw(output='mpl')
        # plt.show()

    def construct_circuit_q(self, diffusion_iterations=1, use_new_checker=False):
        """
        this method acts the same as in "gate" optimization version.
        It constructs a circuit based on the fact, that we can save qubits from edge detection, and instead use
        1 extra ancilla qbit that will save calculations temporarily.

        :param diffusion_iterations: the number of oracle/diffusion calls to achieve probability of
            finding solutions satisfactory
        """
        if diffusion_iterations < 1:
            raise RuntimeError('Diffusion iteration step count cannot be lower than 1')
        self.circuit = QuantumCircuit(
            self.node_qbit_register,  # self.edge_qbit_register, <- this is not used here
            self.quantum_adder_register, self.ancilla_qbit_register,
            self.results_register
        )
        self.circuit.h(self.node_qbit_register)

        adder_subcircuits_collection, diffusion_circuit, condition_check_circuit = self.assemble_subcircuits()

        self.circuit.barrier()
        for _ in range(diffusion_iterations):
            # oracle step
            # circuits in the collection come in tuples with additional information on which qubits it should
            # be glued with into main circuit
            # [
            #     edge_0_qbit, edge_1_qbit, ancilla_qbit,
            #     *self.quantum_adder_register[:adder_gates_required[0].num_qubits-1]
            # ]
            for edge_index, edge, subcircuit in adder_subcircuits_collection:
                self.circuit.compose(subcircuit, [
                    self.node_qbit_register[edge[0]],
                    self.node_qbit_register[edge[1]],
                    self.ancilla_qbit_register[0],
                    *self.quantum_adder_register[:floor(log2(edge_index + 1)) + 1]
                ], inplace=True)
            self.circuit.barrier()

            if not use_new_checker:
                self.circuit.compose(
                    condition_check_circuit,
                    [*self.quantum_adder_register, *self.ancilla_qbit_register],
                    inplace=True)
                self.circuit.z(self.ancilla_qbit_register[1])
                self.circuit.compose(
                    condition_check_circuit.reverse_ops(),
                    [*self.quantum_adder_register, *self.ancilla_qbit_register],
                    inplace=True
                )
            else:
                # new sub-circuit creating already has kickback step propagation through reverse ops baked in
                self.circuit.compose(
                    condition_check_circuit,
                    [*self.quantum_adder_register, *self.ancilla_qbit_register],
                    inplace=True)

            self.circuit.barrier()

            for edge_index, edge, subcircuit in adder_subcircuits_collection[::-1]:
                self.circuit.compose(subcircuit.reverse_ops(), [
                    self.node_qbit_register[edge[0]],
                    self.node_qbit_register[edge[1]],
                    self.ancilla_qbit_register[0],
                    *self.quantum_adder_register[:floor(log2(edge_index + 1)) + 1]
                ], inplace=True)

            # diffusion step
            self.circuit.barrier()
            self.circuit.compose(diffusion_circuit, [*self.node_qbit_register], inplace=True)

        self.circuit.barrier()
        self.circuit.measure(qubit=self.node_qbit_register, cbit=self.results_register)

    def schedule_job_locally(self, shots=1000, seed_simulator=None):
        """
        run circuit measurements locally on your PC with standard settings

        default simulator to use is 'qasm' that provides only counts and measurements, but any can be used
        :returns: job results
        """
        job = AerSimulator().run(self.circuit, shots=shots, seed_simulator=seed_simulator)
        self.current_job_shots = shots
        self.counts = job.result().get_counts(self.circuit)
        return self.counts

    def solve(self, shots: int = 1000, seed_simulator: int = None, verbose: bool = None, diffusion_iterations=1):
        """go to easy method for triggering a solver, locally on your machine"""
        if shots is None:
            raise RuntimeError('shots is a mandatory parameter for this task')
        if shots <= 0:
            raise RuntimeError("shots number must be an integer greater than 0")
        self.diffusion_steps = diffusion_iterations
        if self.optimization == "gates":
            self.construct_circuit_g(
                diffusion_iterations=diffusion_iterations, use_new_checker=self.allow_experimental_runs)
        elif self.optimization == "qbits":
            self.construct_circuit_q(
                diffusion_iterations=diffusion_iterations, use_new_checker=self.allow_experimental_runs)
        self.schedule_job_locally(shots=shots, seed_simulator=seed_simulator)
        if verbose is None:
            verbose = False
        self.solution_analysis(verbose=verbose)

    def check_answers(self, sorted_answers: list):
        """check if any of the answers obtained in "counts" is real solution to the problem posed"""
        self.possible_answers = []
        for proposal in sorted_answers:
            # since the answers from qiskit are ordered in reverse to how we check, reverse sequence and then check
            proposal_ = ("".join(reversed(proposal[0])), proposal[1])
            color_matches = 0
            for edge in self.edge_list:
                if proposal_[0][edge[0]] == proposal_[0][edge[1]]:
                    color_matches += 1

            if self.condition == "=":
                allowed_edges = len(self.edge_list) - self.cuts_number
                if color_matches ^ allowed_edges:
                    continue
                self.possible_answers.append(proposal)
            else:
                if self.min_range <= len(self.edge_list) - color_matches <= self.max_range:
                    self.possible_answers.append(proposal)

        if self.possible_answers:
            if len(self.possible_answers) == len(sorted_answers):
                self.answer_percentage = 100
                return
            good_shots = sum([a[1] for a in self.possible_answers])
            self.answer_percentage = good_shots/self.current_job_shots
            return
        self.answer_percentage = 0

    def solution_analysis(self, verbose=False):
        """
        among given list of answers and counts related to them, search for possible answers of the given problem

        here I get this by checking if every possibility fits the cuts number posed in the problem by counting
        them by hand, meanwhile gathering simple statistics (like "good answer count percentage" and such)

        this function is costly and naive, if you really want to gauge the performance of your solution,
        you might look at something called "quantum counting algorithm", or "quantum phase estimation algorithm"
        """

        if self.counts:
            sorted_answers = sorted([(ans, self.counts[ans]) for ans in self.counts], key=lambda x: x[1])[::-1]
            self.check_answers(sorted_answers)

            # special case of no answers will be visible by both solution and best rejected having
            # still "None" assigned as variables.
            self.answer_percentage = 0.0 if self.possible_answers is [] \
                else round(sum([a[1] for a in self.possible_answers]) / self.current_job_shots, 3) * 100

            if verbose:
                # special case - no answer with current criteria
                if not self.possible_answers:
                    print('Could not find answers with current criteria.\nModify experiment variables if you think '
                          'graph should yield an answer to current problem, or make sure that solution space does not '
                          'take exactly half of the search space in your case.')
                    return

                print(f'there are potentially {len(self.possible_answers)} answers to the graph problem')

                if self.answer_percentage < 2:
                    print(f'in real scenario, without the meaningful number of shots, '
                          f'it is possible that your experiment might not yield a correct answer. Boost number of '
                          f'grover diffusion steps. Experiment might drown true answers in QC uncertainty '
                          f'without any error correction mechanisms.')
                    print('Solutions total number of counts make <2% of total experiment counts.')
                    return
                print(f'possible answers counts are {self.answer_percentage}% of entire experiment counts')
                if self.answer_percentage < 30:
                    print(f'Try running an algorithm with more optimal diffusion step count to boost '
                          f'the total percentage of counts of possible answers to any meaningful level, '
                          f'if this is simulation')

            return

        raise RuntimeError('counts not found, did you actually run algorithm?')


if __name__ == '__main__':
    nodes = 10
    edges = [[1, 9], [4, 5], [2, 8], [3, 5], [1, 3], [0, 9], [2, 9],
             [5, 9], [1, 8], [0, 4], [2, 3], [2, 4], [8, 9], [5, 8], [1, 6], [1, 7]]

    print(f"{len(edges)=}")

    matrix_form = [
        [0 for _ in range(nodes)] for _ in range(nodes)
    ]
    for i, j in edges:
        matrix_form[i][j] = 1

    # initialize through list
    cut = Graph2Cut(nodes, edges, cuts_number=len(edges)-4, optimization="qbits")
    cut.solve(shots=10000, diffusion_iterations=1, seed_simulator=100)

    # initialize through numpy matrix
    print(np.array(matrix_form))
    matrix_cut = Graph2Cut(nodes, edges=np.array(matrix_form), cuts_number=len(edges)-4, optimization="qbits")
    matrix_cut.solve(shots=10000, diffusion_iterations=1, seed_simulator=100)

    matrix_cut_ge = Graph2Cut(
        nodes, edges=np.array(matrix_form), cuts_number=len(edges)-7, optimization="qbits",
        condition=">=", allow_experimental_runs=True)
    matrix_cut_ge.solve(shots=10000, diffusion_iterations=1, seed_simulator=100)
    circ_test = matrix_cut_ge._complex_condition_checking()

    circ_test.draw(output='mpl')
    plt.show()

    print(len(matrix_cut_ge.possible_answers))
    print(len(matrix_cut_ge.counts))

    assert matrix_cut.size() == cut.size()
    assert matrix_cut.possible_answers == cut.possible_answers
    cut.size(as_dict=False)
    print("\nproposed answers:\n", matrix_cut.possible_answers)

