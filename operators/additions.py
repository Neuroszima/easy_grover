from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Qubit
from qiskit.circuit.library.standard_gates.x import XGate, CCXGate

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

        if isinstance(value_register, QuantumRegister):
            self.value_register = value_register
        else:
            self.value_register = QuantumRegister(bits=value_register)

        if isinstance(target_register, QuantumRegister):
            self.target_register = target_register
        else:
            self.target_register = QuantumRegister(bits=target_register)

        self.xgate_lib = []

        for i in range(len(self.target_register)):
            self.xgate_lib.append(XGate().control(i+1))

        super().__init__()

    def _initialize_circuit(self):
        self.circuit = QuantumCircuit(self.value_register, self.target_register)

        # create addition pyramid, from value register to target register

        for v_index, v_qbit in enumerate(self.value_register):
            for t_index in range(len(self.target_register)):
                print(f"appended gate: {(end_gate := -t_index-v_index-1)}")
                print(f"last qbit: {(end_qbit := len(self.target_register)-t_index)}")
                if abs(end_gate) > len(self.target_register):
                    continue
                if t_index != 0:
                    self.circuit.append(
                        instruction=self.xgate_lib[end_gate],
                        qargs=[v_qbit, *self.target_register[v_index:end_qbit]]
                    )
                else:
                    self.circuit.append(
                        instruction=self.xgate_lib[end_gate],
                        qargs=[v_qbit, *self.target_register[v_index:]]
                    )


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

