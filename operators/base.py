from collections import OrderedDict
from copy import deepcopy
from typing import Optional, Literal
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
    def __init__(self, target_register: QuantumRegister | list[Qubit], negate_outcome=False):
        warn("this is an experimental circuit, that is still subject to many changes, use other methods "
             "to achieve results", category=ExperimentalWarning)

        self.target_register = target_register
        self.circuit: QuantumCircuit | None = None
        self.negate_outcome = negate_outcome
        self._gates_total: OrderedDict | None = None
        self._qbits_total: int | None = None
        self._initialize_circuit()

    def __call__(
        self, external_circuit: QuantumCircuit, *, inplace=False, **kwargs  # qbits=None,
    ) -> Optional["QuantumCircuit"]:
        """
        perform circuit composition when invoking call on a class object
        you can specify which qbits should be the target for circuit application by passing them into 'qbits' param
        """
        if inplace:
            external_circuit.compose(self.circuit, qubits=self.target_register, inplace=inplace, inline_captures=True)  # qubits=qbits,
        else:
            # following should not be problematic in smaller cases; in huge cases of several
            # hundred instances however deepcopy might fail due to memory occupation (maybe? idk?).
            # We do this to have completely separate instance
            e_c = deepcopy(external_circuit)
            return e_c.compose(self.circuit, qubits=self.target_register, inplace=inplace, inline_captures=True)  # qubits=qbits,

    @abstractmethod
    def _initialize_circuit(self):
        """override this to define the behaviour of operator and create construction chain"""
        raise NotImplementedError("implement this function to properly instantiate Operator objects")

    def size(self, as_dict=False, target_only_as_linker=False):
        """
        provides a method to evaluate future circuit performance, counting base
        gates that make up this operator circuit, as well as summary of qbits used

        :param as_dict: instead of printing out contents, return dictionary for further processing
        """
        if isinstance(self.circuit, QuantumCircuit):
            pass_manager = PassManager(Unroll3qOrMore(basis_gates=['cx', 'u3']))
            substitution = pass_manager.run(self.circuit)
            self._gates_total: OrderedDict = substitution.count_ops()

        if target_only_as_linker:
            regs_names = [param for param in dir(self) if ("_register" in param) and (param != "target_register")]
            regs_obj = [getattr(self, name_) for name_ in regs_names]
            self._qbits_total = sum([
                len(register)if isinstance(register, (list, QuantumRegister)) else 0
                for register in regs_obj
            ])
        else:
            registers = [
                getattr(self, name) for name in [
                    param for param in dir(self) if "_register" in param
            ]]
            self._qbits_total = sum([
                len(register) if isinstance(register, (list, QuantumRegister)) else 0
                for register in registers])

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

    @staticmethod
    def _negate_register(
        register: QuantumRegister, integer_repr: int, negation_marker: Optional[Literal['0', '1']] = None,
        # skipped_qbits: list[None] | None = None
    ) -> QuantumCircuit:
        """
        Perform negation on the register, based on the integer passed to this method

        The qbits selected for negation are selected based on "0"s in the binary representation of the integer. This
        is the behaviour that reflects preparation of the registers, prior to condition checking. You can also
        prepend XGates on "1" when used as "selected_place_marker"

        :param negation_marker: "0" or "1", by default behaves as "0" selected
        :param register: quantum register, that XGates have to be applied on
        :param integer_repr: integer for a condition check to happen against, or just a "recipe" for gate application
        :returns: circuit ready to compose into bigger structure
        """
        # skipped qbits may be used in future
        circuit = QuantumCircuit(register)

        if negation_marker is None:
            negation_marker = "0"

        if negation_marker not in ["0", "1"]:
            raise ValueError("only valid values for markers are: (\"0\", \"1\")")

        condi_repr = bin(integer_repr)[2:].zfill(len(register))

        for index, c in enumerate(reversed(condi_repr)):
            if c == negation_marker:
                circuit.x(index)

        return circuit


class CircuitBook(ABC):
    """
    Base class that could serve as platform for operators, that are based on applying several smaller circuits,
    either in a row immediately, or interleaved (with, lets say, conditional checks)

    When initialized, the "circuits" field is populated based on the master rule with sub-circuits that are ready
    to be composed into bigger solution.
    """
    def __init__(self):
        warn("this is an experimental circuit, that is still subject to many changes, use other methods "
             "to achieve results", category=ExperimentalWarning)

        self.circuits: list[QuantumCircuit] | None = None
        self._gates_total: OrderedDict | None = None
        self._qbits_max: int | None = None
        self.__current_circuit_index = 0  # for iteration purposes
        self._initialize_circuits()

    def __getitem__(self, item: int):
        return self.circuits[item]

    def __len__(self):
        return len(self.circuits)

    def __reversed__(self):
        return reversed(self.circuits)

    def __call__(
        self, external_circuit: QuantumCircuit, *, inplace=False, index=None, **kwargs
    ) -> Optional["QuantumCircuit"]:
        """
        Use this when you are sure you want to either simply apply a single gate out of the book (provide index if
        you want), or when you want to apply every single sub-circuit one-by-one, however at once
        """
        if isinstance(index, int):
            if inplace:
                external_circuit.compose(self.circuits[index], inplace=inplace)
                return
            else:
                return external_circuit.compose(self.circuits[index], inplace=inplace)
        else:  # assume all circuits to be applied from the book:
            if inplace:
                for circuit in self.circuits:
                    external_circuit.compose(circuit, inplace=inplace)

        # if all false:
        # following should not be problematic in smaller cases; in huge cases of several hundred instances however
        # deepcopy might fail due to memory occupation (maybe? idk?). We do this to have completely separate instance
        e_c = deepcopy(external_circuit)
        for circuit in self.circuits:
            e_c = e_c.compose(circuit, inplace)
        return e_c

    @abstractmethod
    def _initialize_circuits(self):
        """override this to define the behaviour of operator and create construction chain"""
        raise NotImplementedError("implement this function to properly instantiate Operator objects")

    def reverse_ops_all(self):
        """
        Return class object with the book that has every single original circuit, except every circuit
        has their operations reversed
        """
        # I debated on providing this method to all the children classes but idk if absolutely every single class
        # has to implement this, so i will leave this decision for a user to make.
        raise NotImplementedError(f"This method is not defined for class {self.__class__}")

    def size(self, as_dict=False):
        """
        provides a method to evaluate complexity of full array of sub-circuits, counting base
        gates that make them up, as well as max qubits usage among the sub-circuits

        :param as_dict: instead of printing out contents, return dictionary for further processing
        """
        if isinstance(self.circuits, list):
            for c in self.circuits:
                pass_manager = PassManager(Unroll3qOrMore(basis_gates=['cx', 'u3']))
                substitution = pass_manager.run(c)
                if self._gates_total is None:
                    self._gates_total: OrderedDict = substitution.count_ops()
                else:
                    g = substitution.count_ops()
                    for key in g:
                        if key in self._gates_total:
                            self._gates_total[key] += g[key]
                        else:
                            self._gates_total = key

        d = {
            "q_bits": self._qbits_max,  # this param has to be set at the runtime
            "base_instruction_count": self._gates_total
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


class GroverOperator(BaseOperator):
    """
    This defines the grover operator, with all the sweetness of inherited class helpers
    """
    def __init__(
        self, num_of_qbits_covered: int | None = None, target_register: QuantumRegister | list[Qubit] | None = None
    ):
        if num_of_qbits_covered:
            # self.quantum_register = QuantumRegister(num_of_qbits_covered)
            register_placeholder = QuantumRegister(num_of_qbits_covered)
        elif target_register:
            if isinstance(target_register, (QuantumRegister, list)):
                register_placeholder = target_register
            # if isinstance(target_register, QuantumRegister):
            #     self.quantum_register = target_register
            # elif isinstance(target_register, list):
            #     self.quantum_register = QuantumRegister(bits=target_register, name='target')
            else:
                raise TypeError(f"register passed to this constructor is of wrong type: {target_register.__class__}")
        else:
            raise InitializationError("You should pass either a num of qbits that operator works over, of a register")

        # self.multicontrolled_xgate = XGate().control(len(self.quantum_register)-1)
        self.multicontrolled_xgate = XGate().control(len(register_placeholder)-1)

        super().__init__(target_register=register_placeholder)

    def _initialize_circuit(self):
        # self.circuit = QuantumCircuit(self.quantum_register)
        #
        # self.circuit.h(self.quantum_register)
        # self.circuit.x(self.quantum_register)
        #
        # # I had troubles with qiskit recognizing n-(c)ZGate, this construct seems to work as an equivalent
        # self.circuit.h(self.quantum_register[-1])
        # self.circuit.append(self.multicontrolled_xgate, [*self.quantum_register])
        # self.circuit.h(self.quantum_register[-1])
        #
        # self.circuit.x(self.quantum_register)
        # self.circuit.h(self.quantum_register)

        self.circuit = QuantumCircuit(self.target_register)

        self.circuit.h(self.target_register)
        self.circuit.x(self.target_register)

        # I had troubles with qiskit recognizing n-(c)ZGate, this construct seems to work as an equivalent
        self.circuit.h(self.target_register[-1])
        self.circuit.append(self.multicontrolled_xgate, [*self.target_register])
        self.circuit.h(self.target_register[-1])

        self.circuit.x(self.target_register)
        self.circuit.h(self.target_register)


class InitializationError(QiskitError):
    pass


class OperatorError(QiskitError):
    pass


if __name__ == '__main__':
    g = GroverOperator(4)
    g.size()
    g.draw(output='mpl')
