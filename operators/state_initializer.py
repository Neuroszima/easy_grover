from random import randint

from qiskit import QuantumRegister, QuantumCircuit, AncillaRegister, ClassicalRegister
from qiskit.circuit import Qubit
from qiskit.circuit.library.standard_gates.x import XGate, CCXGate
from qiskit.quantum_info import Statevector

from numpy import array, sqrt

from base import BaseOperator, InitializationError


class ExclusionStateInitializer(BaseOperator):
    """
    When selecting qbtis initially to populate states for the experiment, sometimes we know in advance, that
    some of these states do not correspond to our wanted problem configuration. We could, instead of just
    applying hadamard gates everywhere, instantiate register, or some qbits, in a way that covers this case.

    This class helps to decide which state or states we do not want to have in our experiment, by instantiating through
    calculated statevector object, that captures only these states that should happen. Every state prepared by this
    method has an equal chance of happening through
    """

    def __init__(self, inits_bits: QuantumRegister | list[Qubit], excluded_states: list[int]):
        if not isinstance(inits_bits, (list, QuantumRegister)):
            raise TypeError("pass a list of Qubits into a constructor")

        for e in inits_bits:
            if not isinstance(e, Qubit):
                raise TypeError(f"all elements of 'bits' has to be of type: Qubit not {e.__class__}")

        if isinstance(excluded_states, list):
            for s in excluded_states:
                if not isinstance(s, int):
                    TypeError("excluded state should be represented by integer number")
        else:
            TypeError(f"excluded states should be of type 'list', not {excluded_states.__class__}")

        self.excluded_states = excluded_states

        super().__init__(target_register=inits_bits)

    def _initialize_circuit(self):
        """
        this class initializes state through statevector, since it is easier to achieve than preparing separate
        methods for every combination of states
        """
        self.circuit = QuantumCircuit(self.target_register)

        list_of_states = array([
            1 if i not in self.excluded_states else 0
            for i in range(2**len(self.target_register))
        ], dtype=complex)
        list_of_states *= 1/(sum(list_of_states))
        list_of_states = sqrt(list_of_states)
        sv = Statevector(list_of_states)

        self.circuit.initialize(sv)


if __name__ == '__main__':
    # prepare initialization for 4 color case, as well as for 8 color
    # 4 color has 1 exclusion, 8 color has 3
    from random import sample
    from matplotlib import pyplot as plt
    from qiskit_aer import AerSimulator
    from matplotlib import pyplot as plt

    exc_4_color = [randint(0, 2)]
    exc_8_color = sample([*range(8)], k=3)

    print(f"node 1 had {exc_4_color} states excluded")
    print(f"node 2 had {exc_8_color} states excluded")
    q_reg = QuantumRegister(5, name="nodes")
    c_reg = ClassicalRegister(len(q_reg))
    circuit = QuantumCircuit(q_reg, c_reg)
    node_1 = QuantumRegister(bits=[q_reg[0], q_reg[1]], name='node-1')  # 4 color node -> 2 qbits reserved
    node_2 = QuantumRegister(bits=[q_reg[2], q_reg[3], q_reg[4]], name='node-2')  # 8 color -> 3 qbits reserved
    exc_4 = ExclusionStateInitializer(inits_bits=node_1, excluded_states=exc_4_color)
    exc_8 = ExclusionStateInitializer(inits_bits=node_2, excluded_states=exc_8_color)

    exc_4(circuit, inplace=True)
    exc_8(circuit, inplace=True)

    exc_8.size()
    exc_4.size()

    circuit.measure(q_reg, c_reg)

    job = AerSimulator().run(circuit, shots=10000)
    counts = job.result().get_counts()

    # assert counts[bin(condi)[2:].zfill(len(condi_reg))] > 1000  # far more counts than others, avg ~310 with no g_op
    print(counts)
    circuit.draw(output='mpl')
    plt.show()


