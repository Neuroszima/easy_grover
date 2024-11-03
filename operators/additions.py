from math import floor, log2

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Qubit, ControlledGate
from qiskit.circuit.library.standard_gates.x import XGate

from base import BaseOperator, InitializationError


class AdditionFullyCovered(BaseOperator):
    """
    Sub-circuit that deals with addition from one register (value) to another (accumulator)
    """
    def __init__(
        self, value_register: QuantumRegister | list[Qubit], target_register: QuantumRegister | list[Qubit],
    ):
        if len(value_register) > len(target_register):
            raise InitializationError(
                "For this sub-circuit creation purposes, value register should not be of greater length than "
                "target register. Considering swapping registers, or implement your own method of "
                "creating addition operator")
        self.value_register = value_register
        self.addition_target_register = target_register

        # if isinstance(value_register, QuantumRegister):
        #     self.value_register = value_register
        # else:
        #     self.value_register = QuantumRegister(bits=value_register)
        #
        # if isinstance(target_register, QuantumRegister):
        #     self.target_register = target_register
        # else:
        #     self.target_register = QuantumRegister(bits=target_register)

        self.xgate_lib = []

        for i in range(len(self.addition_target_register)):
            self.xgate_lib.append(XGate().control(i+1))

        super().__init__(target_register=[*self.value_register, *self.addition_target_register])

    def _initialize_circuit(self):
        self.circuit = QuantumCircuit(self.value_register, self.addition_target_register)

        # create addition pyramid, from value register to target register

        for v_index, v_qbit in enumerate(self.value_register):
            for t_index in range(len(self.addition_target_register)):
                print(f"appended gate: {(end_gate := -t_index-v_index-1)}")
                print(f"last qbit: {(end_qbit := len(self.addition_target_register)-t_index)}")
                if abs(end_gate) > len(self.addition_target_register):
                    continue
                if t_index != 0:
                    self.circuit.append(
                        instruction=self.xgate_lib[end_gate],
                        qargs=[v_qbit, *self.addition_target_register[v_index:end_qbit]]
                    )
                else:
                    self.circuit.append(
                        instruction=self.xgate_lib[end_gate],
                        qargs=[v_qbit, *self.addition_target_register[v_index:]]
                    )


class AccumulateSingleBitConditions(BaseOperator):
    """
    This sub-circuit can be used as a base for gate-optimized Grover algorithm for graph coloring

    This sub-circuit uses an array of qbits, that serve as a source of uncorrelated flags, and adds them
    one by one into an accumulator register. This can then serve for different checks
    """
    def __init__(
        self, flags_register: QuantumRegister | list[Qubit], accumulator_register: QuantumRegister | list[Qubit],
    ):
        """
        :param flags_register: register containing all the conditionals that can be binary 0 or 1
        :param accumulator_register: target receiving all additions operations
        """

        # if isinstance(flags_register, QuantumRegister):
        #     self.flags_register = flags_register
        # else:
        #     self.flags_register = QuantumRegister(bits=flags_register)
        #
        # if isinstance(accumulator_register, QuantumRegister):
        #     self.accumulator_register = accumulator_register
        # else:
        #     self.accumulator_register = QuantumRegister(bits=accumulator_register)

        self.accumulator_register = accumulator_register
        self.flags_register = flags_register

        self.xgate_lib = []

        for i in range(len(self.accumulator_register)):
            self.xgate_lib.append(XGate().control(i+1))

        super().__init__(target_register=[*self.flags_register, *self.accumulator_register])

    def _initialize_circuit(self):
        self.circuit = QuantumCircuit(self.flags_register, self.accumulator_register)

        all_gates = []
        for qbit_index, qbit in enumerate(self.flags_register):

            # there needs to be a gate pyramid created for each qbit, with control complexity
            # that match floor(log2) of exact qbit in a series to be added.

            temp = floor(int(log2(qbit_index+1))) + 1
            gates_for_qbit = []
            while temp > 0:
                gates_for_qbit.append([qbit, self.xgate_lib[temp]])
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
                control_adder_qbits = [self.accumulator_register[i] for i in range(control_adder_qbits_count)]
            self.circuit.append(
                gate, [qbit, *control_adder_qbits, self.accumulator_register[target_adder_bit]]
            )


class AdditionRepeater(BaseOperator):
    """
    This sub-circuit can be used as a base for qbit-optimized Grover algorithm for graph coloring

    This sub-circuit latches on a single qbit, that serve as a source of flag checks, and adds them
    one by one into an accumulator register. This can then serve for different checks

    This is not a usual operator, as instead of single circuit
    """


if __name__ == '__main__':
    from qiskit import ClassicalRegister
    from qiskit_aer import AerSimulator
    from matplotlib import pyplot as plt
    dummy_val_reg = QuantumRegister(3)
    dummy_target_reg = QuantumRegister(4)
    measure = ClassicalRegister(len(dummy_target_reg))
    starter_circ_q = QuantumCircuit(dummy_val_reg, dummy_target_reg, measure)
    # test val = 6
    starter_circ_q.x(dummy_val_reg[1])
    starter_circ_q.x(dummy_val_reg[2])
    # target accumulator in state -> 7 -->>  7 + 6 = 13 measure result
    starter_circ_q.x(dummy_target_reg[0])
    starter_circ_q.x(dummy_target_reg[1])
    starter_circ_q.x(dummy_target_reg[2])
    starter_circ_q.barrier()

    adder = AdditionFullyCovered(dummy_val_reg, dummy_target_reg)
    starter_circ_q = adder(starter_circ_q)
    starter_circ_q.barrier()
    starter_circ_q.measure(dummy_target_reg, measure)

    # test
    job = AerSimulator().run(starter_circ_q, shots=1000)
    counts = job.result().get_counts()
    print(counts)

    assert counts == {"1101": 1000}
    # 13 in bin -> 0b1101  if MSB is from right-to-left direction

    adder.size()
    adder.draw(output='mpl')
    starter_circ_q.draw(output='mpl')
    plt.show()

