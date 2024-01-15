# import time
# from itertools import pairwise
from itertools import pairwise
from pprint import pprint
from typing import Optional
from math import floor, log2, ceil
from time import perf_counter

# import qiskit.circuit.library
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_aer.backends import AerSimulator
from qiskit.circuit import ControlledGate
from qiskit.circuit.library.standard_gates import XGate
from qiskit.circuit.quantumregister import Qubit


class Graph2Cut:

    ALLOWED_OPTIMIZERS = ["gates", "qbits"]
    ALLOWED_CONDITIONS = ["="]

    def __init__(self, nodes: int, edge_list: list[tuple[int] | list[int]], cuts_number: int = None,
                 condition: str = None, optimization: str = None):
        """
        Perform a graph splitting, based on the graph coloring problem. Only 2 colors are supported by this solver

        For graph splitting, most essential information comes from nodes and edge list, followed
        by the number of cuts to be performed

        "optimization" parameter here makes a tradeoff between number of qbits solution uses and gates.
        "qbits" entry gets rid of entire "edge flagging" part of quantum algorithm, making this up with the
        use of 1 ancilla to store the result temporarily.

        Note: determine the notation order in which answers are served and understand its meaning

        :param nodes: total number of graph members
        :param edge_list: list of 2-member tuples/iterables, that represents graph structure
        :param cuts_number: len(edge_list) is the default
        :param condition: "=" is default
        :param optimization: "gates" or "qbits" are possible modifiers ("gates" is default)
        """
        self.graph_nodes = nodes
        self.edge_list = edge_list
        self.cuts_number = len(self.edge_list) if cuts_number is None else cuts_number
        if condition not in [*self.ALLOWED_CONDITIONS, None]:
            raise NotImplementedError("other types of comparisons other than '=' are not supported yet")
        self.condition = "=" if condition is None else condition
        if optimization in [*self.ALLOWED_OPTIMIZERS, None]:
            self.optimization = "gates" if optimization is None else optimization
        else:
            raise NotImplementedError(f'optimization: {optimization} not recognized as valid optimization')
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
        regiters = [
            self.node_qbit_register, self.edge_qbit_register,
            self.quantum_adder_register, self.ancilla_qbit_register
        ]
        self._qbits_total = sum([len(register) if register else 0 for register in regiters])
        self._gates_total = 0
        self.possible_answers: Optional[list[tuple[str, int]]] = None
        self.diffusion_steps = 0

    def size(self):
        return {
            "c_bits": self.results_register.size,
            "q_bits": self._qbits_total,
            "instruction_count": "unable to calculate"  # self._gates_total
        }

    def _minimal_adder_size(self):
        return floor(log2(len(self.edge_list)))+1

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
        (one of qbits will flip the node qbit state with CX into |1>,
        while the other, being |0>, will not reverse it into |0> again)

        This part only triggers when selected optimization is "gates"
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
        condition_check_circuit = self._condition_checking()
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

    def construct_circuit_g(self, diffusion_iterations=1):
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

    def construct_circuit_q(self, diffusion_iterations=1):
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

            self.circuit.compose(
                condition_check_circuit, [*self.quantum_adder_register, *self.ancilla_qbit_register], inplace=True)
            self.circuit.z(self.ancilla_qbit_register[1])
            self.circuit.compose(
                condition_check_circuit.reverse_ops(),
                [*self.quantum_adder_register, *self.ancilla_qbit_register],
                inplace=True
            )
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
            raise RuntimeError("shots number must be an integer grater than 0")
        self.diffusion_steps = diffusion_iterations
        if self.optimization == "gates":
            self.construct_circuit_g(diffusion_iterations=diffusion_iterations)
        elif self.optimization == "qbits":
            self.construct_circuit_q(diffusion_iterations=diffusion_iterations)
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
            allowed_edges = len(self.edge_list) - self.cuts_number
            color_matches = 0
            for edge in self.edge_list:
                if proposal_[0][edge[0]] == proposal_[0][edge[1]]:
                    color_matches += 1

            if color_matches ^ allowed_edges:
                continue
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

    cut = Graph2Cut(nodes, edges, cuts_number=len(edges)-5, optimization="qbits")
    cut.solve(shots=10000, diffusion_iterations=1)
    cut.solution_analysis(verbose=False)
    print(cut.counts)
    print("\nproposed answers:\n", cut.possible_answers)

    # solution_dictionairy = {}
    # max_iteration_attemps = 8
    # max_nodes = 10
    # solution_dictionairy['optimization'] = 'qbits'
    # solution_dictionairy['global_solver_seed'] = 10
    # print(f'global solver seed: {solution_dictionairy["global_solver_seed"]}')
    #
    # for j in range(3, max_nodes+1):
    #     # ring graph constructor
    #     current_nodes = j
    #     edges = [*pairwise([*range(current_nodes)])]
    #     edges += [(edges[0][0], edges[-1][-1])]
    #     print(f"\n\n\nring: {edges}")
    #     max_allowed_uncut_edges = ceil(j/2)
    #     solution_dictionairy[f"{current_nodes}_ring"] = dict()
    #     for k in range(max_allowed_uncut_edges+1):
    #         solution_dictionairy[f"{current_nodes}_ring"][f"{len(edges) - k}_cuts"] = dict()
    #
    #         for i in range(max_iteration_attemps):
    #             start = perf_counter()
    #             cut = Graph2Cut(current_nodes, edges, cuts_number=len(edges) - k, optimization="qbits")
    #             cut.solve(shots=10000, diffusion_iterations=i+1,
    #                       seed_simulator=solution_dictionairy['global_solver_seed'])
    #             cut.solution_analysis(verbose=False)
    #             end = perf_counter()
    #             # print("shots", cut.current_job_shots)
    #             # print("answers", cut.possible_answers)
    #             # print("rejected", cut.best_rejected_answer)
    #             # print("percentage", cut.answer_percentage)
    #             result_dict = {
    #                 'perf_counter_algo_time': end - start,
    #                 'percentage': cut.answer_percentage,
    #             }
    #             if cut.possible_answers:
    #                 result_dict['answer_count'] = len(cut.possible_answers)
    #                 result_dict['example_answer'] = cut.possible_answers[0]
    #             else:
    #                 result_dict['answer_count'] = 0
    #                 result_dict['example_answer'] = ("", 0)
    #             solution_dictionairy[f"{current_nodes}_ring"][f"{len(edges)-k}_cuts"][f'{i+1}_diff_count'] = result_dict
    #
    # pprint(solution_dictionairy)
    #
    # # select decent answers to a problem
    # for ring in solution_dictionairy:
    #     if ring in ['global_solver_seed', 'optimization']:
    #         continue
    #     for cuts in solution_dictionairy[ring]:
    #         next_cuts = False
    #         # print(cuts, ring)
    #         for diffusion_count in solution_dictionairy[ring][cuts]:
    #             case = solution_dictionairy[ring][cuts][diffusion_count]
    #             if case['percentage'] > 0.1:
    #                 print(ring, cuts, diffusion_count, case['answer_count'])
    #                 next_cuts = True
    #                 break
    #         if next_cuts:
    #             continue
