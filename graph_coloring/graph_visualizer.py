from copy import deepcopy
from math import sqrt, sin, cos, pi
from typing import Optional, Any, Dict

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.spines import Spine
from matplotlib.text import Text
from graph_coloring.circuit_constructor import Graph2Cut

DIM_TYPE = list[int, int] | tuple[int, int]
SCALE_TYPE = list[float, float] | tuple[float, float]
plt.rcParams['font.family'] = 'monospace'


def rgb_to_matlab(r, g, b) -> tuple[float, float, float]:
    """
    converts regular intensity representation to MATLAB-like format, handled by matplotlib
    """
    return r/255., g/255., b/255.


class Graph2CutVisualizer:
    """
    graph solution visualization tool for graph solver

    you can treat this visualizer as separate tool that does not need a graph solver (by passing desired parameters
    directly as values), or you may pass an entire solver instance to visualize results
    """

    def __init__(self, nodes: Optional[int] = 0, edge_list: list[tuple[int, ...] | list[int, ...]] = None,
                 solutions: Optional[list] = None, graph_solver: Optional[Graph2Cut] = None,
                 base_graph_size: Optional[float] = 5.):
        self.solutions = solutions if solutions else None
        if nodes and edge_list:
            self.nodes = nodes
            self.edges = edge_list
            self.results = None
            self.shots = None
        elif graph_solver:
            self.nodes = graph_solver.graph_nodes
            self.edges = graph_solver.edge_list
            self.results = graph_solver.counts
            self.solutions = graph_solver.possible_answers
            self.cuts = graph_solver.cuts_number
            self.shots = graph_solver.current_job_shots
        else:
            raise ValueError("Neither nodes accompanying list of edges, nor a graph solver instance has been "
                             "passed. Cannot initialize visualizer.")
        # self.graph_image_size: DIM_TYPE = (1000, 1000)
        self.screen_diagonal = 23
        self.inner_outer_scale_edge = 0.25
        self.node_radius = 10
        self.edge_length = 75
        self.free_drawing_space = 50
        self.base_graph_size = base_graph_size

    @property
    def inner_graph_area(self):
        """defines proportions of the entire image that would influence where hub nodes and triples get drawn."""
        return [self.inner_outer_scale_edge, 1-self.inner_outer_scale_edge]

    @property
    def left_outer_graph_area(self):
        """defines proportions of the entire image that would influence where single nodes and some pairs get drawn."""
        return [0, self.inner_outer_scale_edge]

    @property
    def right_outer_graph_area(self):
        """same as left outer area, sets"""
        return [1-self.inner_outer_scale_edge, 1]

    @staticmethod
    def flatten_once(arr: list):
        """flatten an array (write all the arrays sub-array contents into a new array)"""
        tmp = []
        for sub_array in arr:
            tmp.extend(sub_array)
        return tmp

    @staticmethod
    def euclidean_distance(point1, point2):
        """calculate distance between the points"""
        return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def draw_nodes_on_circumference(self, node_map: dict, nodes_count: int, middle_point: tuple | list | None = None):
        """spread all the nodes in a circular fashion around, then keep track of which lines to draw"""
        middle_point_ = middle_point if middle_point else [500, 500]
        angles = [i*360/nodes_count for i in range(nodes_count)]
        if nodes_count == 4:
            distance = self.edge_length/sqrt(2)
        elif nodes_count == 3:
            distance = self.edge_length * sqrt(3)/2
        elif nodes_count == 2:
            distance = self.edge_length/2
        elif nodes_count > 4:
            # to evenly spread out more nodes around we will use couple angles and minimal distance
            gamma = (180 - (360/nodes_count))/2
            gamma_p = 90 - gamma
            distance = self.edge_length * cos(pi * gamma_p / 180) / cos(pi * (gamma - gamma_p) / 180)
            print("in draw_circ.; values:")
            print(gamma, gamma_p, distance)
        else:
            raise ValueError(f"improper number of nodes: {nodes_count}")

        for node, angle in zip(range(nodes_count), angles):
            node_map[node]["coordinates"] = (
                middle_point_[0] + distance * cos(pi * angle / 180),
                middle_point_[1] + distance * sin(pi * angle / 180)
            )

        # following keeps track of edges that have to be displayed on the graph image, due to nodes being shown
        # otherwise, we could as well just copy edges as an output which would be useless
        lines = []
        d_edges = deepcopy(self.edges)
        for node in range(nodes_count):
            if node_map[node]["coordinates"] != [None, None]:
                to_add = [e for e in d_edges if node in e]
                for e in to_add:
                    d_edges.remove(e)
                lines.extend(to_add)

        return node_map, lines

    def draw_generic_graph(self, ax: Axes, fig: Figure, node_map: dict):
        fig.suptitle(f"Graph structure, n={len(node_map.keys())}")
        for node in node_map:
            if node_map[node]['coordinates'] != [None, None]:
                ax.add_artist(Circle(
                    xy=node_map[node]['coordinates'],
                    radius=self.node_radius,
                    facecolor=rgb_to_matlab(255, 255, 255),
                    edgecolor=rgb_to_matlab(0, 0, 0),
                    zorder=5
                ))
                ax.add_artist(Text(
                    x=node_map[node]['coordinates'][0],
                    y=node_map[node]['coordinates'][1],
                    text=f'{node_map[node]["index"]}',
                    color=rgb_to_matlab(0, 0, 0),
                    verticalalignment='center',
                    horizontalalignment='center',
                    zorder=10
                ))
        spines = ['bottom', 'top', 'left', 'right']
        ax.spines: Dict[str, Spine]  # noqa
        for spine in spines:
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

        return ax, fig

    def draw_graph_solution(
            self, ax: Axes, fig: Figure, node_map: dict,
            solution_number: int = 0, top: int = None, left: int = None):
        """
        it draws only one example solution. If you want to draw a particular one, pass a number os parameter
        additionally, this method draws only solution of 2 color case
        """
        text_color = rgb_to_matlab(0, 0, 0)
        yellow_partition_member = rgb_to_matlab(255, 255, 102)
        green_partition_member = rgb_to_matlab(153, 255, 102)
        fig.suptitle(f"Graph solution, edges={len(self.edges)}, Cut_count={self.cuts}")
        solution = [*self.results.keys()][solution_number]

        # the "byte ordering" of the qiskit counts is reversed compared to how we need to start painting nodes
        for node_index, c_bit_repr in enumerate(solution[::-1]):
            if c_bit_repr == "1":
                c_color = yellow_partition_member
            else:
                c_color = green_partition_member
            ax.add_artist(Circle(
                xy=node_map[node_index]['coordinates'],
                radius=self.node_radius,  # * len(node_map[chunk]['contents'])
                color=c_color,
                zorder=5
            ))
            ax.add_artist(Text(
                x=node_map[node_index]['coordinates'][0],
                y=node_map[node_index]['coordinates'][1],
                text=f'{node_map[node_index]["index"]}',
                color=text_color,
                verticalalignment='center',
                horizontalalignment='center',
                zorder=10
            ))
        if top and left:
            counts = self.results[solution]
            # p = "NaN" if not self.shots else counts/self.shots
            ax.add_artist(Text(
                x=left,
                y=top-10,
                text=f'counts for this solution: {counts}',  # , percentage: {round(p,3)}%
                color=text_color,
                verticalalignment='center',
                horizontalalignment='left',
                zorder=10
            ))

            if tuple([solution, counts]) in self.solutions:
                display_text = 'Good Solution'
            else:
                display_text = 'Bad Solution'

            ax.add_artist(Text(
                x=left,
                y=top-25,
                text=display_text,
                color=text_color,
                verticalalignment='center',
                horizontalalignment='left',
                zorder=10
            ))

        spines = ['bottom', 'top', 'left', 'right']
        ax.spines: Dict[str, Spine]  # noqa
        for spine in spines:
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelleft=False, left=False)
        ax.tick_params(labelbottom=False, bottom=False)

        return ax, fig

    def draw_graph(
            self, present_solution=False, select_good=False,
            selected_answer: Optional[int] = None, save_params: dict | None = None):
        """
        Draw edges and nodes on the graph

        Prior to this method rework, there was second drawing method available; the only one available now is on
        circle circumference.

        You can now however pass "save_params" as dict with "image_name" (str type) and "image_dpi" (int type)
        to trigger saving the figure as PNG.
        """
        nodes_with_connections = [[node, set(self.flatten_once([edge for edge in self.edges if node in edge]))]
                                  for node in range(self.nodes)]

        for n in nodes_with_connections:
            n[1].remove(n[0])
            n.append(len(n[1]))

        sorted_node_tab = sorted(nodes_with_connections, key=lambda element: -element[2])

        node_map: dict[int, dict[str, Any]] = {
            e[0]: {
                "index": e[0],
                "connected_nodes": e[1],
                "conn_count": e[2],
                "coordinates": [None, None],
            } for e in sorted_node_tab
        }

        node_map, lines = self.draw_nodes_on_circumference(node_map, nodes_count=len(nodes_with_connections))

        # for each part that would contain a piece of content, draw circle and text with number of nodes in it
        # the smaller the circle the fewer nodes it represents
        # keyword list: "contents" (list[list]), "coordinates" (x,y), "position": (inner, outer)
        min_y, max_y = 250, 750
        min_x, max_x = 250, 750

        window_init = True
        for node in node_map:
            if node_map[node]['coordinates'] != [None, None]:
                if window_init:
                    min_x, max_x = [node_map[node]['coordinates'][0]] * 2
                    min_y, max_y = [node_map[node]['coordinates'][1]] * 2
                    window_init = False
                    continue
                if node_map[node]['coordinates'][0] > max_x:
                    max_x = node_map[node]['coordinates'][0]
                if node_map[node]['coordinates'][0] < min_x:
                    min_x = node_map[node]['coordinates'][0]
                if node_map[node]['coordinates'][1] > max_y:
                    max_y = node_map[node]['coordinates'][1]
                if node_map[node]['coordinates'][1] < min_y:
                    min_y = node_map[node]['coordinates'][1]

        min_y -= self.free_drawing_space
        min_x -= self.free_drawing_space
        max_y += self.free_drawing_space
        max_x += self.free_drawing_space

        x_span = max_x - min_x
        y_span = max_y - min_y
        ratio = x_span/y_span

        if ratio > 1:
            figsize = (ratio * self.base_graph_size, self.base_graph_size)
        else:
            ratio = 1/ratio
            figsize = (self.base_graph_size, ratio * self.base_graph_size)

        fig, ax = plt.subplots(figsize=figsize)
        fig: Figure
        ax: Axes
        plt.subplots_adjust(**{
            "left": 0.03, "right": 0.97,
            "bottom": 0, "top": 0.92,
        })

        ax.set_aspect(1)
        ax.set_ylim(min_y, max_y)
        ax.set_xlim(min_x, max_x)

        # draw nodes
        if present_solution and self.solutions:
            if selected_answer:
                ax, fig = self.draw_graph_solution(
                    ax, fig, node_map, solution_number=selected_answer, top=max_y, left=min_x)
            elif select_good:
                sorted_answers = sorted([(ans, self.results[ans]) for ans in self.results], key=lambda x: x[1])[::-1]
                print(sorted_answers)
                print(self.solutions)
                g_index = [*self.results.keys()].index(sorted_answers[0][0])
                ax, fig = self.draw_graph_solution(
                    ax, fig, node_map, solution_number=g_index, top=max_y, left=min_x)
            else:
                ax, fig = self.draw_graph_solution(ax, fig, node_map, top=max_y, left=min_x)
        else:
            ax, fig = self.draw_generic_graph(ax, fig, node_map)

        # draw lines
        for line in lines:
            ax.add_artist(Line2D(
                xdata=[node_map[p]["coordinates"][0] for p in line],
                ydata=[node_map[p]["coordinates"][1] for p in line],
                linewidth=2,
                color='black',
                marker='None',
                zorder=0
            ))

        if isinstance(save_params, dict):
            fig.savefig(
                fname=str(save_params.get("image_name", "example"))+".png",
                dpi=save_params.get("image_dpi", 300)
            )
        else:
            plt.show()


if __name__ == '__main__':
    nodes_ = 10
    edges = [[1, 9], [4, 5], [2, 8], [3, 5], [1, 3], [0, 9], [2, 9],
             [5, 9], [1, 8], [0, 4], [2, 3], [2, 4], [8, 9], [5, 8], [1, 6], [1, 7]]

    solver = Graph2Cut(nodes_, edges, cuts_number=len(edges)-6, optimization='qbits')
    solver.solve(shots=10000, diffusion_iterations=1)
    solver.solution_analysis()

    visualiser = Graph2CutVisualizer(graph_solver=solver)
    visualiser.draw_graph(present_solution=True, select_good=True)

