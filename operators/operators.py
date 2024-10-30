from qiskit import QuantumRegister, QuantumCircuit, AncillaRegister
from qiskit.circuit import Qubit
from qiskit.circuit.library.standard_gates.x import XGate, CCXGate

from base import BaseOperator, InitializationError


class EqualityOperator3Ancilla(BaseOperator):
    """
    class that provides atomic construction of sub-circuits realising equality/inequality operator

    this operator allocates without the need to call separate 'allocate_qbits' method like Graph2Cut or similar do

    this operator uses 3 ancilla, 2 for storing separate, specific number checks, and then the 3rd to store the actual
    equality check. For now this should suffice, however target should use only 1 ancilla to do the job

    ancilla register however can be reused later down the circuit.
    """
    def __init__(
        self, numbers_checked: list, compare_register_size: int | None = None,
        first_register: QuantumRegister | list[Qubit] | None = None,
        second_register: QuantumRegister | list[Qubit] | None = None,
        result_storage_register: QuantumRegister | list[Qubit] | Qubit | None = None, negate_outcome=False
    ):

        # first initialization method - only int passed
        if all([not first_register, not second_register, not result_storage_register]) and compare_register_size:
            if isinstance(compare_register_size, int):
                self.first_register = QuantumRegister(compare_register_size, name="first")
                self.second_register = QuantumRegister(compare_register_size, name="second")
                self.ancilla_register = AncillaRegister(3)
            else:
                raise TypeError(f"selected compare_register_size init is of invalid type: "
                                f"{type(compare_register_size)}, should be 'int'")

        # second init - registers have been passed or lists of Qubits (for example -> unpacked registers)
        elif all([first_register, second_register, result_storage_register]) and (not compare_register_size):
            if isinstance(first_register, QuantumRegister):
                self.first_register = first_register
            else:
                self.first_register = QuantumRegister(bits=first_register, name="first")
            if isinstance(second_register, QuantumRegister):
                self.second_register = second_register
            else:
                self.second_register = QuantumRegister(bits=second_register, name="second")

            if isinstance(result_storage_register, QuantumRegister):
                self.ancilla_register = result_storage_register
            elif isinstance(result_storage_register, Qubit):
                self.ancilla_register = QuantumRegister(bits=[result_storage_register])
            else:
                self.ancilla_register = QuantumRegister(bits=result_storage_register)

        # wrong initialization
        else:
            raise InitializationError("This class should initialize either through passing 'int', or every"
                                      "register that would be target / store data.")

        if len(self.first_register) != len(self.second_register):
            raise InitializationError("This operator only serves equal length registers comparison")

        self.numbers_checked: list = numbers_checked

        # we only need one instantiated gate like this
        self.multicontrolled_xgate = XGate().control(len(self.first_register))
        super().__init__(negate_outcome=negate_outcome)

    def _initialize_circuit(self):
        print("inside constructor")
        print(f"i am calling from {self.__class__} class object")

        chr_nums = [bin(num)[2:].zfill(len(self.first_register)) for num in self.numbers_checked]
        # check the compatibility woth the length of the registers
        for n in chr_nums:
            if len(n) > len(self.first_register):  # locked sizes of registers to eq len
                raise ValueError(f'binary repr. : {n} is too long for this check to be performed')

        self.circuit = QuantumCircuit(self.first_register, self.second_register, self.ancilla_register)

        # when inversing, we could append X-gate now or later, doesn't matter. Lets do it here:
        if self.negate_outcome:
            self.circuit.x(self.ancilla_register[-1])

        for chr_num in chr_nums:
            # calculate condition
            for i, c in enumerate(reversed(chr_num)):
                if c == '0':  # in both registers we have to have the same bits on same positions
                    self.circuit.x(self.first_register[i])
                    self.circuit.x(self.second_register[i])
            self.circuit.append(self.multicontrolled_xgate, [*self.first_register, self.ancilla_register[0]])
            self.circuit.append(self.multicontrolled_xgate, [*self.second_register, self.ancilla_register[1]])

            self.circuit.append(CCXGate(), [*self.ancilla_register])  # save end res in last ancilla

            # reclaiming resources
            self.circuit.append(self.multicontrolled_xgate, [*self.first_register, self.ancilla_register[0]])
            self.circuit.append(self.multicontrolled_xgate, [*self.second_register, self.ancilla_register[1]])
            for i, c in enumerate(reversed(chr_num)):
                if c == '0':
                    self.circuit.x(self.first_register[i])
                    self.circuit.x(self.second_register[i])


class GroverOperator(BaseOperator):
    """
    This defines the grover operator, with all the sweetness of inherited class helpers
    """
    def __init__(
        self, num_of_qbits_covered: int | None = None, operated_register: QuantumRegister | list[Qubit] | None = None
    ):
        if num_of_qbits_covered:
            self.quantum_register = QuantumRegister(num_of_qbits_covered)
        elif operated_register:
            if isinstance(operated_register, QuantumRegister):
                self.quantum_register = operated_register
            elif isinstance(operated_register, list):
                self.quantum_register = QuantumRegister(bits=operated_register)
            else:
                raise TypeError(f"register passed to this constructor is of wrong type: {operated_register.__class__}")
        else:
            raise InitializationError("You should pass either a num of qbits that operator works over, of a register")

        self.multicontrolled_xgate = XGate().control(len(self.quantum_register)-1)

        super().__init__()

    def _initialize_circuit(self):
        self.circuit = QuantumCircuit(self.quantum_register)

        self.circuit.h(self.quantum_register)
        self.circuit.x(self.quantum_register)

        # I had troubles with qiskit recognizing n-(c)ZGate, this construct seems to work as an equivalent
        self.circuit.h(self.quantum_register[-1])
        self.circuit.append(self.multicontrolled_xgate, [*self.quantum_register])
        self.circuit.h(self.quantum_register[-1])

        self.circuit.x(self.quantum_register)
        self.circuit.h(self.quantum_register)


if __name__ == '__main__':
    eq = EqualityOperator3Ancilla([0, 1, 2], 2, negate_outcome=True)

    eq.size()
    eq.draw(output='mpl')

    g = GroverOperator(4)
    g.size()
    g.draw(output='mpl')
