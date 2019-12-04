"""Utilities for representing pose graphs
"""

import random
import math
from math import sqrt
import numpy as np
import quaternion
import sophus as sp


def in_range(x, range_):
    return x >= range_[0] and x <= range_[1]


def get_upper_triangle(matrix):
    """Get the upper triangular part of the matrix
    Return:
        numpy 1D array containing the upper triangle part
    """
    M, N = matrix.shape
    assert(M == N)
    return matrix[np.triu_indices(M)]


class Quaternion:
    """Convenience wrapper around quaternion library
        `quaternion` follows the convention (w, x, y, z)
        this wrapper transforms to (x, y, z, w)
    """
    def __init__(self, q):
        self.q = np.asarray(q)

    @staticmethod
    def from_R(R):
        quat = quaternion.from_rotation_matrix(R)
        comp = quat.components
        quat = np.hstack( (comp[1:], comp[:1]) )
        return Quaternion(quat)

    @staticmethod
    def to_R(q):
        """
        Args:
            q: numpy array in (x, y, z, w) form
        """
        q_copy = np.asarray(q)
        quat = np.hstack( (q_copy[3:], q_copy[:3]) )
        qu = quaternion.from_float_array(quat)
        return quaternion.as_rotation_matrix(qu)


class Node2D:
    """Node of a 2D graph, representing a pose

    Attributes:
        id_: node id
        x: x position
        y: y position
        theta: angle (TODO: confirm angle is from x-axis)
    """
    def __init__(self, id_, x, y, theta):
        self.id_ = id_
        self.x = x
        self.y = y
        self.theta = theta

    def to_g2o(self):
        """Return a string representing the node in g2o format
        """
        return "VERTEX_SE2 {} {} {} {}".format(self.id_, self.x, self.y,
                                               self.theta)

    def pose(self):
        return self.x, self.y, self.theta


class Node3D:
    """Node of a 3D graph, representing a pose
    """
    def __init__(self, id_, pos, quat):
        """
        Args:
            pos: position of node
            quat: orientation of node, represented by quaternion (x, y, z, w)
        """
        self.id_ = id_
        self.t = np.asarray(pos)
        self.q = np.asarray(quat)

    def to_g2o(self):
        """Return a string representing the node in g2o format
        """
        line = "VERTEX_SE3:QUAT {} ".format(self.id_)
        line += " ".join([str(x) for x in self.t])
        line += " ".join([str(x) for x in self.q])
        return line

    def pose(self):
        R = Quaternion.to_R(self.q)
        T = np.identity(4)
        T[:3, :3] = R
        T[:3, 3] = self.t
        return T

class Edge2D:
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
        assert(len(info) == 6)
        self.i = i
        self.j = j
        self.x = x
        self.y = y
        self.theta = theta
        self.info = info

    def to_g2o(self):
        """Return a string representing the edge in g2o format
        """
        line = "EDGE_SE2 {} {} {} {} {} ".format(self.i, self.j, self.x, self.y,
                                               self.theta)
        line += " ".join([str(x) for x in self.info])
        return line

    def measurement(self):
        return self.x, self.y, self.theta

    def diagonal_sigmas(self):
        return 1/sqrt(self.info[0]), 1/sqrt(self.info[3]), 1/sqrt(self.info[5])

    def has_diagonal_info(self):
        return self.info[1] == self.info[2] == self.info[4] == 0

    def __str__(self):
        return "{} {} {} {} {} {}".format(self.i, self.j, self.x, self.y, self.theta,
                                   self.info)

    def __repr__(self):
        return str(self)

class Edge3D:
    """Edge of a 3D graph, representing a measurement between 2 nodes
    """
    def __init__(self, i, j, t, q, info):
        assert(len(info) == 21)
        self.i = i
        self.j = j
        self.t = np.asarray(t)
        self.q = np.asarray(q)
        self.info = np.asarray(info)

    def to_g2o(self):
        """Return a string representing the edge in g2o format
        """
        line = "EDGE_SE3:QUAT {} {} ".format(self.i, self.j)
        line += " ".join([str(x) for x in self.t]) + " "
        line += " ".join([str(x) for x in self.q]) + " "
        line += " ".join([str(x) for x in self.info])
        return line

    def info_mat(self):
        """
        Return:
            information matrix as 2D numpy array
        """
        N = 6
        info_mat = np.zeros(shape=(N, N))
        start = 0
        for i in range(N):
            info_mat[i, i:] = self.info[start: start + N-i]
            start += N-i
        info_mat = info_mat + info_mat.T - np.diag(info_mat.diagonal())
        assert(np.allclose(info_mat, info_mat.T))
        return info_mat

    def cov(self):
        return np.linalg.inv(self.info_mat())

    def measurement(self):
        R = Quaternion.to_R(self.q)
        T = np.identity(4)
        T[:3, :3] = R
        T[:3, 3] = self.t
        return T

class SingleRobotGraph:
    """Single robot graph representation of g2o file

    Attributes:
        nodes: nodes of the graph
        odom_edges: odometry edges
        loop_closure_edges: loop closure edges
    """
    def __init__(self):
        self.nodes = {}
        self.odom_edges = {}
        self.loop_closure_edges = {}

    def read_from(self, fpath):
        """Read the graph from g2o file

        Args:
            fpath: input g2o file path
        """
        with open(fpath) as fp:
            line = fp.readline()
            while line:
                self._process_line(line)
                line = fp.readline()

    def print_summary(self):
        """Print summary of the graph
        """
        print("# Nodes: {}".format(len(self.nodes)))
        print("# Odometry edges: {}".format(len(self.odom_edges)))
        print("# Loop closure edges: {}".format(len(self.loop_closure_edges)))


    def write_to(self, fpath):
        """Write graph to file
        """
        fp = open(fpath, "w+")
        for node in self.nodes.values():
            fp.write(node.to_g2o() + "\n")
        for edge in list(self.odom_edges.values()) + list(
                       self.loop_closure_edges.values()):
            fp.write(edge.to_g2o() + "\n")


class SingleRobotGraph2D(SingleRobotGraph):
    """2D single robot graph
    """
    def _process_line(self, line):
        """Read in a single line of 2D g2o file as node or edge

        Args:
            line: a line of 2D g2o file

        Raises:
            Exception: The line does not start with a known tag
        """
        values = line.split()
        tag = values[0]
        if tag == "VERTEX_SE2":
            id_ = int(values[1])
            x, y, theta = [float(v) for v in values[2:]]
            self.nodes[id_] = Node2D(id_, x, y, theta)
        elif tag == "EDGE_SE2":
            i, j = [int(x) for x in values[1:3]]
            x, y, theta = [float(v) for v in values[3:6]]
            info = [float(v) for v in values[6:]]
            edge = Edge2D(i, j, x, y, theta, info)
            if abs(i-j) == 1:
                self.odom_edges[(i, j)]  = edge
            else:
                self.loop_closure_edges[(i, j)] = edge
        else:
            raise Exception("Line with unknown tag")

    def to_multi(self, n_max_inter_lc=15):
        """Extract a multi-robot graph from current graph

        Returns:
            A multi-robot graph
        """
        multi_graph = MultiRobotGraph2D()
        multi_graph.read_nodes(self.nodes)
        multi_graph.read_edges(self.odom_edges, self.loop_closure_edges,
                               n_max_inter_lc)
        return multi_graph


class SingleRobotGraph3D(SingleRobotGraph):
    def _process_line(self, line):
        """Read in a single line of 3D g2o file as node or edge

        Args:
            line: a line of 3D g2o file

        Raises:
            Exception: The line does not start with a known tag
        """
        values = line.split()
        tag = values[0]
        if tag == "VERTEX_SE3:QUAT":
            id_ = int(values[1])
            pos = [float(v) for v in values[2:5]]
            quat = [float(v) for v in values[5:]]
            self.nodes[id_] = Node3D(id_, pos, quat)
        elif tag == "EDGE_SE3:QUAT":
            i, j = [int(x) for x in values[1:3]]
            translation = [float(v) for v in values[3:6]]
            quat = [float(v) for v in values[6:10]]
            info = [float(v) for v in values[10:]]
            edge = Edge3D(i, j, translation, quat, info)
            if abs(i-j) == 1:
                self.odom_edges[(i, j)]  = edge
            else:
                self.loop_closure_edges[(i, j)] = edge
        else:
            raise Exception("Line with unknown tag")

    def to_multi(self, n_max_inter_lc=15):
        """Extract a multi-robot graph from current graph

        Returns:
            A multi-robot graph
        """
        multi_graph = MultiRobotGraph3D()
        multi_graph.read_nodes(self.nodes)
        multi_graph.read_edges(self.odom_edges, self.loop_closure_edges,
                               n_max_inter_lc)
        return multi_graph


class MultiRobotGraph:
    """Multi robot graph

    Attributes:
        N: number of robots
        nodes: nodes for each robot
        odoms: odometry edges for each robot
        lc: loop closure edges for each robot
        inter_lc: loop closure edges between robots
    """

    def __init__(self):
        self.N = 2
        self.nodes = [{} for _ in range(self.N)]
        self.odoms = [{} for _ in range(self.N)]
        self.lc = [{} for _ in range(self.N)]
        self.inter_lc = {}

        # meta info
        self.ranges = [[] for _ in range(self.N)]

    def read_from(self, fpath):
        """Read multi robot graph from g2o file

        Args:
            fpath: input g2o file path
        """
        with open(fpath) as fp:
            line = self._process_meta(fp)
            while line:
                self._process_line(line)
                line = fp.readline()

    def _process_meta(self, fp):
        """Process meta info of multi robot g2o
        """
        line = fp.readline()
        while line.startswith("#Robot"):
            values = line.split()
            robot_idx = int(values[1])
            start = int(values[2])
            end = int(values[3])
            self.ranges[robot_idx] = [start, end]
            line = fp.readline()
        return line

    def read_nodes(self, nodes):
        """Split single robot nodes into nodes for 2 robots
        """
        segment_len = (len(nodes)+1)//self.N
        for k, v in nodes.items():
            idx = k // segment_len
            assert(idx < self.N)
            self.nodes[idx][k] = v

    def read_edges(self, odom_edges, loop_closure_edges, n_max_inter_lc=15):
        """Split single robot edges into edges for 2 robots
        """
        for odom in odom_edges.values():
            for idx, nodes in enumerate(self.nodes):
                if odom.i in nodes and odom.j in nodes:
                    i, j = odom.i, odom.j
                    self.odoms[idx][(i, j)] = odom
                    break

        inter_lc = []
        for lc in loop_closure_edges.values():
            is_self_lc = False
            for idx, nodes in enumerate(self.nodes):
                if lc.i in nodes and lc.j in nodes:
                    i, j = lc.i, lc.j
                    self.lc[idx][(i, j)] = lc
                    is_self_lc = True
                    break
            if not is_self_lc:
                inter_lc.append(lc)

        # Randomly choose n_max_inter_lc of inter robot lc
        if n_max_inter_lc < len(inter_lc):
            inter_lc = random.sample(inter_lc, n_max_inter_lc)

        for lc in inter_lc:
            i, j = lc.i, lc.j
            self.inter_lc[(i, j)] = lc

    def write_to(self, fpath):
        """Write graph to file
        """
        fp = open(fpath, "w+")
        # Write meta info on robots nodes separation
        start = 0
        for i, nodes in enumerate(self.nodes):
            length = len(nodes)
            end = start + length - 1
            fp.write("#Robot {} {} {}\n".format(i, start, end))
            start = end + 1

        for nodes in self.nodes:
            for node in nodes.values():
                fp.write(node.to_g2o() + "\n")
        for edges in self.odoms + self.lc + [self.inter_lc]:
            for edge in edges.values():
                fp.write(edge.to_g2o() + "\n")

    def print_summary(self):
        """Print summary of the multi robot graph
        """
        print("# Robots: {}".format(self.N))
        print("# Nodes: " + " ".join([str(len(x)) for x in self.nodes]))
        print("# Odoms: " + " ".join([str(len(x)) for x in self.odoms]))
        print("# Inner loop closures: " + " ".join([str(len(x))
                                                    for x in self.lc]))
        print("# Inter loop closures: {}".format(len(self.inter_lc)))

    def add_perceptual_aliasing_lc(self, M=2, N=5):
        """Add perceptual aliasing loop closures

        Args:
            M: number of groups of aliases
            N: number of loop closures in each group
        """
        pass # TODO Implement this

    def to_singles(self):
        """Convert multi robot graph into separate single robot graphs

        Returns:
            A list of single robot graphs
        """
        single_graphs = []
        for i in range(self.N):
            graph = SingleRobotGraph()
            graph.nodes = self.nodes[i]
            graph.odom_edges = self.odoms[i]
            graph.loop_closure_edges = self.lc[i]
            single_graphs.append(graph)
        return single_graphs

    def set_inter_lc(self, inter_lc):
        """Set the inter robot lc from the lc list
        Args:
            inter_lc: list of inter loop closures
        """
        self.inter_lc = {}
        for edge in inter_lc:
            i, j = edge.i, edge.j
            self.inter_lc[(i, j)] = edge

    def merge_to_single(self):
        """Merge the multi robot graph as a single robot graph

        Returns:
            A single robot graph
        """
        graph = SingleRobotGraph()
        for i in range(self.N):
            graph.nodes.update(self.nodes[i])
            graph.odom_edges.update(self.odoms[i])
            graph.loop_closure_edges.update(self.lc[i])
        graph.loop_closure_edges.update(self.inter_lc)
        return graph


class MultiRobotGraph2D(MultiRobotGraph):
    """2D Multi robot graph
    """
    def add_random_inter_lc(self, N=20):
        """Add randomly generated inter loop closures

        TODO: Specify noise values, rather than hard-coded
        """
        x_mu = random.uniform(-5, 5)
        x_sigma = 0.15
        y_mu = random.uniform(-5, 5)
        y_sigma = 0.15
        theta_mu = random.uniform(-math.pi, math.pi)
        theta_sigma = 0.15

        info = [1/x_sigma**2, 0, 0, 1/y_sigma**2, 0, 1/theta_sigma**2]

        random_inter_lc = [Edge2D(random.choice(list(self.nodes[0])),
                                random.choice(list(self.nodes[1])),
                                random.normalvariate(x_mu, x_sigma),
                                random.normalvariate(y_mu, y_sigma),
                                random.normalvariate(theta_mu, theta_sigma),
                                info)
                            for _ in range(N)]
        random_inter_lc = {(edge.i, edge.j) : edge for edge in random_inter_lc}
        self.inter_lc.update(random_inter_lc)

    def _process_line(self, line):
        """Process g2o line
        """
        values = line.split()
        tag = values[0]
        if tag == "VERTEX_SE2":
            id_ = int(values[1])
            x, y, theta = [float(v) for v in values[2:]]
            for idx, range_ in enumerate(self.ranges):
                if in_range(id_, range_):
                    self.nodes[idx][id_] = Node2D(id_, x, y, theta)
                    break
        elif tag == "EDGE_SE2":
            i, j = [int(x) for x in values[1:3]]
            x, y, theta = [float(v) for v in values[3:6]]
            info = [float(v) for v in values[6:]]
            edge = Edge2D(i, j, x, y, theta, info)

            found = False
            for idx, nodes in enumerate(self.nodes):
                if i in nodes and j in nodes:
                    if abs(i-j) == 1:
                        self.odoms[idx][(i, j)] = edge
                    else:
                        self.lc[idx][(i, j)] = edge
                    found = True
                    break
            if not found:
                self.inter_lc[(i, j)] = edge
        else:
            raise Exception("Line with unknown tag")


class MultiRobotGraph3D(MultiRobotGraph):
    """3D Multi robot graph
    """
    def add_random_inter_lc(self, N=20):
        """Add randomly generated inter loop closures

        TODO: Specify noise values, rather than hard-coded
        """
        # Random translation specification
        t_mu = [random.uniform(-5, 5) for _ in range(3)]
        t_sigma = [1, 1, 1]

        # Random rotation specification
        r_mu = [random.uniform(-math.pi, math.pi) for _ in range(3)]
        r_sigma = [0.5, 0.5, 0.5]

        # Information matrix
        info_mat = np.diag([1/x**2 for x in t_sigma + r_sigma])
        info = get_upper_triangle(info_mat)

        random_inter_lc = {}
        for _ in range(N):
            i = random.choice(list(self.nodes[0]))
            j = random.choice(list(self.nodes[1]))
            t = [random.gauss(mu, sigma) for mu, sigma in zip(t_mu, t_sigma)]

            # Get random quaterion
            r = [random.gauss(mu, sigma) for mu, sigma in zip(r_mu, r_sigma)]
            R = sp.SO3.exp(np.array(r)).matrix()
            q = Quaternion.from_R(R).q

            random_inter_lc[(i, j)] = Edge3D(i, j, t, q, info)

        self.inter_lc.update(random_inter_lc)
