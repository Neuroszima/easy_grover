from copy import deepcopy
from math import sqrt, sin, cos, pi
from typing import Optional, Any, Dict, Literal
from warnings import warn

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.spines import Spine
from matplotlib.text import Text
from circuit_constructor import Graph2Cut

DIM_TYPE = list[int, int] | tuple[int, int]
SCALE_TYPE = list[float, float] | tuple[float, float]
plt.rcParams['font.family'] = 'monospace'


def rgb_to_matlab(r, g, b) -> tuple:
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
        self.graph_image_size: DIM_TYPE = [1000, 1000]
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

    def spread_surrounding_nodes(self, node_map, current_node: int):
        """
        calculate all the angles at which nodes should be spread outwards from the node in focus,
        while taking into account the node we came from, if there is need for that
        """
        # first, get count of how many other nodes spread even further out, to the nodes that we want to
        # draw in this step apply a correction, if the nodes already selected in this very step are
        # somewhat connected to each other

        # the "adj_node" express exactly same members as the "surrounding_node", but we take them into account while
        # counting from different perspectives, that's why we need 2 different methods of iterations and "2 scopes"
        # one is for counting from the perspective of the center node that we try to spread from (adj_node), and the other
        # is when we take account connectivity of the surrounding nodes between themselves (not accounting center then)

        # so we kind of do double-duty check here
        surrounding_elements_count = []
        for adj_node in node_map[current_node]["connected_nodes"]:
            count = node_map[adj_node]["conn_count"]
            correction = sum([
                1 if surrounding_node in node_map[adj_node]["connected_nodes"] else 0
                for surrounding_node in node_map[current_node]["connected_nodes"]
            ])
            surrounding_elements_count.append(count - correction)

        s = sum(surrounding_elements_count)

        # spread out the unaccounted nodes - calc. angles of spread, and later obtain coordinates of the graph members
        if s > 1:
            # calculate allowed space size and start angle that will get passed to spread out nodes visually
            if not node_map[current_node]["spreadout_origin_angle"] \
                    and not node_map[current_node]["spreadout_origin_angle"]:
                allowed_angle_span = 360
                start_angle = 0
            else:
                allowed_angle_span = 360 - node_map[current_node]["spreadout_reserved_angle"]
                start_angle = node_map[current_node]["spreadout_origin_angle"] + \
                              node_map[current_node]["spreadout_reserved_angle"] / 2
                if start_angle > 360:
                    start_angle -= 360

            partition = [round(allowed_angle_span * part / s) for part in surrounding_elements_count]
            angle_origins = [start_angle + sum(partition[:index])
                             for index, element in enumerate(partition)]
            angles = [p / 2 + o for p, o in zip(partition, angle_origins)]
        else:
            partition = []
            angles = [180]

        # set other nodes in the map for future spreading
        if len(partition) >= 2:
            origin = node_map[current_node]["coordinates"]
            for adj_node, angle, part in zip(node_map[current_node]["connected_nodes"], angles, partition):
                node_map[adj_node]["coordinates"] = [
                    origin[0] + round(self.edge_length * sin(pi * angle / 180)),
                    origin[1] + round(self.edge_length * cos(pi * angle / 180))
                ] if node_map[adj_node]["coordinates"] == [None, None] else node_map[adj_node]["coordinates"]
                # node_map[adj_node]["spreadout_origin_angle"] = angle
                if (origin_angle := 180 + angle) < 360:
                    node_map[adj_node]["spreadout_origin_angle"] = origin_angle
                else:
                    node_map[adj_node]["spreadout_origin_angle"] = origin_angle - 360
                if 180 - part < 0:
                    node_map[adj_node]["spreadout_reserved_angle"] = 60
                else:
                    node_map[adj_node]["spreadout_reserved_angle"] = 180 - part

        return node_map

    def draw_lines_from_node(self, node_map: dict, current_node: int):
        """
        set the coordinates of the nodes with respect to the node in focus, and then, after the connected nodes
        coordinate pairs are finally known, output a list of lines (point-point pairs) to be drawn in the graph image.

        First node always starts in the center, and connected nodes spread out around the node in focus. Furthermore,
        space that is left for other nodes (in terms of the "free area" that will be left, surrounding the nodes,
         that spread out of the one in focus) is weighted, accounting for the adjacent nodes that also can have
        connections themselves.

        For example:
            We start from the node 3, which has 5 nodes that are connected to it.
            If the graph is a star, and nodes have no additional connection, other than this one (which is node nr.3),
            each one of them will be treated with equal weight, and spread evenly around. The spread mechanism will use
            the circle as the base, and each adjacent node "coordinate" will originate on the circle (more or less - keep
            rounding error in mind)

        Another example:
            We start from node nr.3, have connections to 5 other nodes, but one of them (lets say, nr.6) has 3 other
            connections. Nodes will be spread, taking into account the node 6 having 3x as high of a weight, and the angle
            between nodes adjacent to node nr. 6 will be wider, and will become more narrow for other nodes.

        In other words: Angle "X36" and "63Y" will be wider in latter case, while in first case all angles made from the
        graph connections will have the same value.

        If the node already has a coordinate, limit the angle of other nodes to be drawn and use allowed angle spread as the
        permitted space to draw nodes that have not been accounted for yet.

        :param node_map: keeps the track of coordinates of entire graph, and interconnectivity information
        :param current_node: current node we focus on spreading outwards from
        :return: updated node_map, list of lines to draw
        """
        warn("this method is experimental")

        def check_node_placement(node_map__: dict, current_node__: int,):  # angles: list
            """
            if node could hide a potential connection (by visually being placed in the middle of another
            connection of 2 different nodes), move it slightly up
            """
            neighbour_count = node_map__[current_node__]["conn_count"]
            if neighbour_count % 2 == 0 and neighbour_count:
                print('node placement check confirmed')
                node_map__[current_node__]["coordinates"] = [
                    node_map__[current_node__]["coordinates"][0],
                    node_map__[current_node__]["coordinates"][1] + self.node_radius
                ]
            return node_map__

        lines = []

        # set the graph origin if this is first node
        if node_map[current_node]["coordinates"] == [None, None]:
            node_map[current_node]["coordinates"] = [500, 500]

        for k in node_map[current_node]["connected_nodes"]:
            if node_map[current_node]["index"] in node_map[k]["connected_nodes"]:
                node_map[k]["connected_nodes"].remove(node_map[current_node]["index"])

        node_map = self.spread_surrounding_nodes(node_map, current_node)

        # check if any connection will have a potential to get hidden when being drawn
        node_map = check_node_placement(node_map, current_node) # angles

        # keep the track of lines that spread from center node
        lines.extend([
            # (node_map[current_node]["coordinates"], node_map[adj_node]["coordinates"])
            # for adj_node in node_map[current_node]["connected_nodes"]
            (current_node, adj_node) for adj_node in node_map[current_node]["connected_nodes"]
        ])

        # add the lines that connect between surrounding nodes
        pairs_to_add = []
        for adj_node in node_map[current_node]["connected_nodes"]:
            for surrounding_node in node_map[adj_node]["connected_nodes"]:
                # if connection between surrounding nodes - add a connection to be drawn
                if surrounding_node in node_map[current_node]["connected_nodes"]:
                    # can't modify contents of arr while running loop so saving for later
                    # prevent duplicate due to mirroring
                    if (
                            ((surrounding_node, adj_node) not in pairs_to_add) and
                            ((adj_node, surrounding_node) not in pairs_to_add)
                    ):
                        pairs_to_add.append((surrounding_node, adj_node))

        for surrounding_node, adj_node in pairs_to_add:
            # lines.append((node_map[surrounding_node]["coordinates"], node_map[adj_node]["coordinates"]))
            lines.append((surrounding_node, adj_node))
            node_map[adj_node]["connected_nodes"].remove(surrounding_node)
            node_map[surrounding_node]["connected_nodes"].remove(adj_node)

        if node_map[current_node]["connected_nodes"]:
            # draw the one that has most connections first
            connected_nodes = [(index, node_map[index]["conn_count"])
                               for index in node_map[current_node]["connected_nodes"]]
            s_connected_nodes = sorted(connected_nodes, key=lambda x: x[1], reverse=True)
            for node_index, _ in s_connected_nodes:
                node_map, lines__ = self.draw_lines_from_node(node_map, node_index)
                lines.extend(lines__)

        # next_nodes = node_map_[current_node["index"]]["connected_nodes"]
        node_map[current_node]["connected_nodes"] = set()
        return node_map, lines  # , next_nodes

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
                    radius=self.node_radius,  # * len(node_map[chunk]['contents'])
                    # color=rgb_to_matlab(255, 255, 255),
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
        ax.tick_params(labelleft=False, left=False)
        ax.tick_params(labelbottom=False, bottom=False)

        return ax, fig

    def draw_graph_solution(
            self, ax: Axes, fig: Figure, node_map: dict,
            solution_number: int = 0, top: int = None, left: int = None):
        """it draws only one example solution. If you want to draw a particular one, pass a number os parameter"""
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
                ax.add_artist(Text(
                    x=left,
                    y=top-25,
                    text=f'Good Solution',
                    color=text_color,
                    verticalalignment='center',
                    horizontalalignment='left',
                    zorder=10
                ))
            else:
                ax.add_artist(Text(
                    x=left,
                    y=top-25,
                    text=f'Bad Solution',
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

    def draw_graph(self, present_solution=False, select_good=False,
                   selected_answer: Optional[int] = None,
                   draw_type: Optional[Literal["map", "circle"]] = None):
        """draw edges and nodes on the graph in possibly the least offensive way..."""
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
                # below a fragment that denotes banned space that needs to be left free for visual purposes
                "spreadout_origin_angle": None,  # at what angle does the previous connection come from?
                "spreadout_reserved_angle": None,  # how much of a angular space does the previous connection take?
            } for e in sorted_node_tab
        }
        current_node_ = sorted_node_tab[0]

        # below, "lines" is completely equivalent to edge list, when graph is drawn fully
        # however we might wish to draw only a part of the graph, and then this will keep track of nodes drawn
        if not draw_type:
            draw_type = "circle"
        if draw_type == "map":
            # this solves node positioning recursively
            node_map, lines = self.draw_lines_from_node(node_map, current_node_[0])
        elif draw_type == "circle":
            node_map, lines = self.draw_nodes_on_circumference(node_map, nodes_count=len(nodes_with_connections))
        else:
            raise ValueError("Improperly selected draw_type")

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

        plt.show()
        # fig.savefig(fname='example_graph_image.png', dpi=300)


if __name__ == '__main__':
    nodes_ = 10
    edges = [[1, 9], [4, 5], [2, 8], [3, 5], [1, 3], [0, 9], [2, 9],
             [5, 9], [1, 8], [0, 4], [2, 3], [2, 4], [8, 9], [5, 8], [1, 6], [1, 7]]

    solver = Graph2Cut(nodes_, edges, cuts_number=len(edges)-6, optimization='qbits')
    solver.solve(shots=10000, diffusion_iterations=1)
    solver.solution_analysis()

    visualiser = Graph2CutVisualizer(graph_solver=solver)
    visualiser.draw_graph(present_solution=True, select_good=True, draw_type="circle")
    # visualiser = Graph2CutVisualizer(nodes=nodes_, edge_list=edges)
    # visualiser.draw_graph(draw_type="circle")

