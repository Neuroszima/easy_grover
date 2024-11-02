from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Qubit
from qiskit.circuit.library.standard_gates.x import XGate

from operators.base import BaseOperator, CircuitBook


class ConditionalPhaseFlipper(BaseOperator):
    """
    Flip the phase of quantum state that fulfills certain equality condition, and reverse the operation to
    propagate through phase kickback

    This class uses n-controlled XGate to check which state should be phase-flipped, with the use of ZGate anchored at
    single qbit ancilla register. The n-controlled XGate checks the case of fulfilling certain integer number, so
    that the phase flip happens only on certain state of the register.

    Circuit made by this operator already has the reversed operations at the end (meaning - conditional, n-controlled
    XGate and XGates that negate the "0"s in integer are already applied also at the end of the circuit in reversed
    order)
    """

    def __init__(
        self, condition: int, condition_register: QuantumRegister | list[Qubit],
        ancilla_register: QuantumRegister | list[Qubit]
    ):
        self.condition = condition
        if isinstance(condition_register, QuantumRegister):
            self.condition_register = condition_register
        else:
            self.condition_register = QuantumRegister(bits=condition_register, name="condi")

        if isinstance(ancilla_register, QuantumRegister):
            self.ancilla_register = ancilla_register
        else:
            self.ancilla_register = QuantumRegister(bits=ancilla_register, name="anc")

        self.cx_gate = XGate().control(len(self.condition_register))

        super().__init__()

    def _initialize_circuit(self):
        self.circuit = QuantumCircuit(self.condition_register, self.ancilla_register)

        neg = self._negate_register(self.condition_register, self.condition)

        self.circuit.compose(neg, inplace=True)
        self.circuit.append(self.cx_gate, [*self.condition_register, *self.ancilla_register])
        self.circuit.z(self.ancilla_register[-1])
        self.circuit.append(self.cx_gate, [*self.condition_register, *self.ancilla_register])
        self.circuit.compose(neg, inplace=True)


class MulticonditionalPhaseFlipper(CircuitBook):
    pass



if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from qiskit_aer import AerSimulator
    from operators.base import GroverOperator
    from qiskit import ClassicalRegister
    anc = QuantumRegister(1)
    condi = 25
    condi_reg = QuantumRegister(len(bin(condi)[2:]))
    results_reg = ClassicalRegister(len(condi_reg))
    circuit = QuantumCircuit(condi_reg, anc, results_reg)

    flipper = ConditionalPhaseFlipper(condition=condi, condition_register=condi_reg, ancilla_register=anc)
    g = GroverOperator(target_register=condi_reg)

    circuit.h(condi_reg)
    circuit.barrier()
    flipper(circuit, inplace=True)
    circuit.barrier()
    g(circuit, inplace=True)
    circuit.measure(condi_reg, results_reg)

    job = AerSimulator().run(circuit, shots=10000)
    counts = job.result().get_counts()

    assert counts[bin(condi)[2:].zfill(len(condi_reg))] > 1000  # far more counts than others, avg ~310 with no g_op
    print(counts)
    circuit.draw(output='mpl')
    plt.show()

