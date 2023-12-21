from math import ceil, sqrt
from random import random
from typing import Optional

from tkinter import Tk
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from matplotlib.patches import Circle
from matplotlib.text import Text
from circuit_constructor import Graph2Cut

DIM_TYPE = list[int, int] | tuple[int, int]
SCALE_TYPE = list[float, float] | tuple[float, float]

root = Tk()
MM_TO_INCH = 10/25.4
SCREEN_WIDTH = root.winfo_screenwidth()
SCREEN_WIDTH_MM = root.winfo_screenmmwidth()
SCREEN_WIDTH_INCH = SCREEN_WIDTH_MM * MM_TO_INCH
SCREEN_HEIGHT = root.winfo_screenheight()
SCREEN_HEIGHT_MM = root.winfo_screenmmheight()
SCREEN_HEIGHT_INCH = SCREEN_HEIGHT_MM * MM_TO_INCH
root.quit()
del root
print(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_HEIGHT_MM, SCREEN_WIDTH_MM)


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
                 solutions: Optional[list] = None, graph_solver: Optional[Graph2Cut] = None):
        self.solutions = solutions if solutions else None
        if nodes and edge_list:
            self.nodes = nodes
            self.edges = edge_list
        elif graph_solver:
            self.nodes = graph_solver.graph_nodes
            self.edges = graph_solver.edge_list
            self.solutions = [s for s in graph_solver.counts]
        else:
            raise ValueError("Neither nodes accompanying list of edges, nor a graph solver instance has been "
                             "passed. Cannot initialize visualizer.")
        self.graph_image_size: DIM_TYPE = [1000, 1000]
        self.screen_diagonal = 23
        self.inner_outer_scale_edge = 0.25
        self.node_radius = 30
        self.edge_length = 20

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

    @property
    def graph_image_size_inches(self):
        """
        gets inch representation of an image size when, originally, one chose width/height in pixels
        helper property to convert into matplotlib-friendly figure values.
        """
        return (
            self.graph_image_size[0] / SCREEN_WIDTH * SCREEN_WIDTH_INCH,
            self.graph_image_size[1] / SCREEN_HEIGHT * SCREEN_HEIGHT_INCH
        )

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

    def show_chunk_placements(self, content_map: dict):
        """helper method to evaluate planned placements, prior to drawing actual nodes"""
        fig, ax = plt.subplots(figsize=self.graph_image_size_inches)
        fig: Figure
        ax: Axes
        # for each part that would contain a piece of content, draw circle and text with number of nodes in it
        # the smaller the circle the fewer nodes it represents
        # keyword list: "contents" (list[list]), "coordinates" (x,y), "position": (inner, outer)

        ax.set_aspect(1)
        ax.set_ylim(0, self.graph_image_size[0])
        ax.set_xlim(0, self.graph_image_size[1])
        for chunk in content_map:
            ax.add_artist(Circle(
                xy=content_map[chunk]['coordinates'],
                radius=self.node_radius*len(content_map[chunk]['contents']),
                color=rgb_to_matlab(0, 255, 255)
            ))
            ax.add_artist(Text(
                x=content_map[chunk]['coordinates'][0],
                y=content_map[chunk]['coordinates'][1],
                text=f'{len(content_map[chunk]["contents"])}',
                color=rgb_to_matlab(255, 0, 0),
                verticalalignment='center',
                horizontalalignment='center'
            ))

        # fig.show()
        # plt.draw()
        plt.show()

    def chunk_nodes(self, nodes_with_attributes: list):
        """
        Chunk nodes that connect together. Try to form a group of three nodes (i.e. find
        instances of one node that has connection to 2 others) based on graph interconnectivity.
        """
        chunks = []
        current_chunk = []
        # sort by number of connections to deal with high-connectivity nodes first
        sorted_node_tab = sorted(nodes_with_attributes, key=lambda element: -element[2])
        j = 0
        while sorted_node_tab:
            # always chunk at least 2 connected nodes
            print("new chunk now", current_chunk)
            current_chunk.append(sorted_node_tab[0])
            current_node = sorted_node_tab.pop(0)
            current_node_connections = current_node[1]
            if not sorted_node_tab:
                # this was the last node in entire graph to take account for
                chunks.append(current_chunk)
                break
            i = 0
            # here we try to form a triple of nodes, we do not check for triangles however
            while len(current_chunk) <= 2 and i < 2:
                print(i)
                to_pop = -1
                for index, n in enumerate(sorted_node_tab):
                    if n[0] in current_node_connections:
                        print("pop found")
                        to_pop = index
                        break
                if to_pop >= 0:
                    print('poping thing:', to_pop)
                    # popping elements found from sorted tab into chunk
                    current_chunk.append(sorted_node_tab.pop(to_pop))
                    print("chunk now", current_chunk)
                # sanity value when we do not find anything useful in node tab, for example a node with
                # single connection to somewhere else.
                i += 1
            j += 1
            chunks.append(current_chunk)
            current_chunk = []
            if j > 10:
                break

        return chunks

    def spread_inner_parts(self, inner_parts_count: int):
        pass

    def spread_outer_parts(self, outer_parts_count, full_space=False):
        pass

    def calculate_chunk_coordinates(self, node_chunks: list[list]):
        """pre-plan initial chunk 'areas' that would occupy target image of a graph"""
        # inner area of a graph should be taken by the triples, especially if those contain hubs, outside area should
        # be either pairs or single edges (logically those can be thrown practically anywhere)
        content_map = {}
        inner_parts = []
        all_checked = False
        while not all_checked:
            selected = -1
            for index, chunk in enumerate(node_chunks):
                if len(chunk) == 3:
                    selected = index
                    inner_parts.append(chunk)
            if selected != -1:
                node_chunks.pop(selected)
                continue

            all_checked = True
        # either all that is left will get converted into outer_parts or only some chunks
        outer_parts = node_chunks

        # obtain decent coordinates that chunks could place themselves into
        if inner_parts and outer_parts:
            legal_inner_positions = self.spread_inner_parts(len(inner_parts))
            legal_outer_positions = self.spread_outer_parts(len(outer_parts), full_space=False)
        else:
            legal_inner_positions = []
            legal_outer_positions = self.spread_outer_parts(outer_parts, full_space=True)

        print(legal_inner_positions)

    def init_canvas(self):
        """plan initial edges and nodes on the graph in the possibly nice and clear way"""
        nodes_with_connections = [[node, set(self.flatten_once([edge for edge in self.edges if node in edge]))]
                                  for node in range(self.nodes)]

        for n in nodes_with_connections:
            n[1].remove(n[0])
            n.append(len(n[1]))

        # decide what is a hub and what isn't a hub - hub could be inside a graph picture where it could be easier to
        # avoid big line overlaps, when there is large number of nodes to connect between
        # an 'average' of how many other nodes single node sees
        hub_factor = sum([n[2] for n in nodes_with_connections])/self.nodes
        print(hub_factor)
        for n in nodes_with_connections:
            if n[-1] >= ceil(hub_factor) + 1:
                n.append(True)
            else:
                n.append(False)

        nodes_chunks = self.chunk_nodes(nodes_with_connections)
        content_map = self.calculate_chunk_coordinates(nodes_chunks)
        # self.show_chunk_placements(content_map)


if __name__ == '__main__':
    nodes_ = 10
    edges = [[1, 9], [4, 5], [2, 8], [3, 5], [1, 3], [0, 9], [2, 9],
             [5, 9], [1, 8], [0, 4], [2, 3], [2, 4], [8, 9], [5, 8], [1, 6], [1, 7]]

    solver = Graph2Cut(nodes_, edges, cuts_number=len(edges) - 3, optimization='qbits')
    solver.solve()
    solver.solution_analysis()

    visualiser = Graph2CutVisualizer(graph_solver=solver)
    # print(visualiser.graph_image_size_inches)
    visualiser.init_canvas()

