"""Convert a single-robot g2o file to a multi-robot g2o file

Example usages:
  $ python3 convert_to_multi.py input.g2o output.g2o
"""

import argparse


class Node:
    """Node of a 2D graph, representing a pose

    Attributes:
        id_: node id
        x: x position
        y: y position
        theta: angle (TODO(Jay): confirm angle is from x-axis)
    """

    def __init__(self, id_, x, y, theta):
        self.id_ = id_
        self.x = x
        self.y = y
        self.theta = theta


class Edge:
    """Edge of a 2D graph, representing a measurement between 2 nodes

    Attributes:
        i: first node id
        j: second node id
        x: x translation measurement
        y: y translation measurement
        theta: angle measurement
        info: information matrix, stored as the upper-triangular block in
              row-major order, in a list
    """
    def __init__(self, i, j, x, y, theta, info):
        self.i = i
        self.j = j
        self.x = x
        self.y = y
        self.theta = theta
        self.info = info


class SingleRobotGraph:
    """Single robot graph representation of g2o file

    TODO(Jay) Currently assumes 2D, extend to 3D

    Attributes:
        nodes: nodes of the graph
        odom_edges: odometry edges
        loop_closure_edges: loop closure edges
    """

    def __init__(self, fpath):
        """Construct the graph from file

        Args:
            fpath: input g2o file path
        """
        self.is_single_robot = True
        self.nodes = {}
        self.odom_edges = []
        self.loop_closure_edges = []
        with open(fpath) as fp:
            line = fp.readline()
            while line:
                self._process_line(line)
                line = fp.readline()

    def _process_line(self, line):
        """Read in a single line of file as node or edge

        Args:
            line: a line of g2o file

        Raises:
            Exception: The line does not start with a known tag
        """
        values = line.split()
        tag = values[0]
        if tag == "VERTEX_SE2":
            id_ = int(values[1])
            x, y, theta = [float(x) for x in values[2:]]
            self.nodes[id_] = Node(id_, x, y, theta)
        elif tag == "EDGE_SE2":
            i, j = [int(x) for x in values[1:3]]
            x, y, theta = [float(x) for x in values[3:6]]
            info = [float(x) for x in values[6:]]
            edge = Edge(i, j, x, y, theta, info)
            if abs(i-j) == 1:
                self.odom_edges.append(edge)
            else:
                self.loop_closure_edges.append(edge)
        else:
            raise Exception("Line with unknown tag")

    def print_summary(self):
        """Print summary of the graph
        """
        print("# Nodes: {}".format(len(self.nodes)))
        print("# Odometry edges: {}".format(len(self.odom_edges)))
        print("# Loop closure edges: {}".format(len(self.loop_closure_edges)))

    def to_multi(self):
        """Extract a multi-robot graph from current graph

        Returns:
            A multi-robot graph
        """
        multi_graph = MultiRobotGraph()
        multi_graph.read_nodes(self.nodes)
        multi_graph.read_edges(self.odom_edges, self.loop_closure_edges)
        # TODO(Jay) complete this

class MultiRobotGraph:
    """Multi robot graph

    TODO(Jay) Currently assumes 2 robots in 2D, extend to many robots in 3D

    Attributes:
        TODO(Jay)
    """

    def __init__(self):
        self.robot_a_nodes = {}
        self.robot_b_nodes = {}
        self.robot_a_odom = []
        self.robot_a_odom = []
        self.robot_a_lc = []
        self.robot_a_lc = []

    def read_nodes(self, nodes):
        """Split single robot nodes into nodes for 2 robots
        """
        for k, v in nodes.items():
            if k < len(nodes)//2:
                self.robot_a_nodes[k] = v
            else:
                self.robot_b_nodes[k] = v

    def read_edges(self, odom_edges, loop_closure_edges):
        """Split single robot edges into edges for 2 robots
        """
        pass # TODO(Jay) Complete this



if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Convert a single-robot g2o file to a multi-robot g2o file")
    parser.add_argument("input_fpath", metavar="input.g2o", type=str,
                        help="input g2o file path")
    parser.add_argument("output_fpath", metavar="output.g2o", type=str,
                        nargs='?', default="output.g2o",
                        help="output g2o file path")
    args = parser.parse_args()

    # Construct graph from g2o file
    graph = SingleRobotGraph(args.input_fpath)

    print("========== Input g2o Graph Summary ================")
    graph.print_summary()
