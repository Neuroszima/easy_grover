from collections import OrderedDict
from typing import Optional
from warnings import warn
from abc import ABC, abstractmethod

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit
from qiskit.exceptions import ExperimentalWarning, QiskitError
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroll3qOrMore
from qiskit.circuit.library.standard_gates.x import XGate

from matplotlib import pyplot as plt


class BaseOperator(ABC):
    """
    class that provides atomic construction of sub-circuits realising equality/inequality operator

    this operator allocates without the need to call separate 'allocate_qbits' method like Graph2Cut or similar do
    """
    def __init__(self, negate_outcome=False):
        warn("this is an experimental circuit, that is still subject to many changes, use other methods "
             "to achieve results", category=ExperimentalWarning)

        self.circuit: QuantumCircuit | None = None
        self.negate_outcome = negate_outcome
        self._gates_total: OrderedDict | None = None
        self._qbits_total: int | None = None
        self._initialize_circuit()

    def __call__(
            self, external_circuit: QuantumCircuit, *, inplace=False, **kwargs
    ) -> Optional["QuantumCircuit"]:

        if inplace:
            external_circuit.compose(self.circuit, inplace=inplace)
        else:
            return external_circuit.compose(self.circuit, inplace=inplace)

    @abstractmethod
    def _initialize_circuit(self):
        """override this to define the behaviour of operator and create construction chain"""
        raise NotImplementedError("implement this function to properly instantiate Operator objects")

    def size(self, as_dict=False):
        """
        provides a method to evaluate future circuit performance, counting base
        gates that make up this operator circuit, as well as summary of qbits used

        :param as_dict: instead of printing out contents, return dictionary for further processing
        """
        if isinstance(self.circuit, QuantumCircuit):
            pass_manager = PassManager(Unroll3qOrMore(basis_gates=['cx', 'u3']))
            substitution = pass_manager.run(self.circuit)
            self._gates_total: OrderedDict = substitution.count_ops()

        regiters = [getattr(self, name) for name in [param for param in dir(self) if "_register" in param]]
        self._qbits_total = sum([len(register) if register else 0 for register in regiters])
        d = {
            "q_bits": self._qbits_total,
            "base_instruction_count": self._gates_total  # unable to calculate
        }
        if as_dict:
            return d
        else:
            for key in d:
                if key == "base_instruction_count" and self._gates_total:
                    print("%s: {" % key)
                    for k_ in d[key]:
                        print(f"    {k_}: {d[key][k_]},")
                    print("}")
                else:
                    print(f"{key}: {d[key]},")

    def draw(self, output='text', save_output=False):
        """callback to draw the main circuit without the need to type all the code"""

        if self.circuit:
            repr_ = self.circuit.draw(output=output)
            print(repr_)
            if not save_output:
                if output == 'mpl':
                    plt.show()
            else:
                pass
        else:
            raise QiskitError("circuit not constructed so no visualization is accessible")

    def reverse_ops(self):
        """reverses base circuits operations"""
        if self.circuit:
            return self.circuit.reverse_ops()
        raise RuntimeError("Circuit does not exist so operation reversion is not possible")


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


class InitializationError(QiskitError):
    pass


class OperatorError(QiskitError):
    pass


if __name__ == '__main__':
    g = GroverOperator(4)
    g.size()
    g.draw(output='mpl')
