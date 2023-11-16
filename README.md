# Easy grover
Everything related to grover search algorithm for quantum computations

A collection of tools designed to automate quantum circuit creation for solving problems described below.
Solutions are made in ``qiskit`` library.

### graph coloring

a subset of problems concentrated around graph structure partitioning. 
Can be useful in partitioning datasets, some problems related to looking at layout differences.
This type of approach also has a connection to traveling salesman problem.

For now automated graph coloring is realised in a fashion of 2-cut algorithm, with simple addition being 
required to count edges connected by nodes with color mismatch. So far implementation only includes a stiff 
edge counting that is compared with a single value. 

##### Possible TODO's

1. Restriction to check against single edge sum can be extended to any bracket. For example solutions that allow edges 
   to signify connections between nodes of the same color can be allowed. In that case, it might be useful to 
   check a couple different values. We might construct a problem that reflects an example of "allow more 
   than 25 cuts to be performed", lets say, on a graph with a total of 30 connections between nodes.
2. Different types of initialization can be performed to lower potentialneed for additional q-bits or save on toffoli 
   complexity of the end circuit.
3. Potential q-bit reuse and optimizations, like usage of different gate combinations to achieve lower running 
   costs on actual quantum computers.
4. Forming a solution in a different library, like ```cirq``` to boost the effectiveness of the algorithm and ease
   compilation of the solution on different quantum computer brands (like Google, IBM, D-Wave and such...)
5. Extending the automation tools for creating circuits to solve a k-cut problem - variable number of colors a 
   graph represents
6. Functions that let one schedule jobs in actual IBM cloud.

### Usage

Typical usage of the library is presented below:

To solve a coloring problem for a graph, you need to prepare a list of pairs that represent a graph structure,
number of nodes, then decide how many edges need to be cut. Only equality operand is supported now, meaning solution 
will only present graph coloring for the matching number of cuts in the graph structure.

```python
from graph_coloring import circuit_constructor

graph_nodes = 3
edges = [(0, 1), (1, 2)]
graph_cutter = circuit_constructor.Graph2Cut(
    nodes=graph_nodes,
    edge_list=edges,
    cuts_number=len(edges),
    condition="="
)
```

Next, you can run a circuit construction procedure. Keep in mind that usually you could min-max probability of 
finding a proper coloring solution. Here you can do this by appropriately choosing number of iterations an oracle 
querry paired up with grover diffusion has to be invoked. Address this by changing ``diffusion_iterations`` parameter, 
then run your circuit.

```
graph_cutter.construct_circuit(diffustion_iterations=2)
# schedule_job_locally runs 'qasm_simulator' type of job on your PC
results = graph_cutter.schedule_job_locally()
print(results)

# you could also use this monstrosity:
# print(sorted([(ans, results[ans]) for ans in results], key=lambda x: x[1])[::-1])
```

Member of an object -> ``graph_cutter.circuit`` has all the gate operations needed for it to be run, so 
one could run it using different simulator or send it to cloud. Alternatively one could also look at its graphical 
representation, though some more complicated designs may not be visualized correctly.