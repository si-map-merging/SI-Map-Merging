"""Utilities for representing pose graphs
"""

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

    def to_g2o(self):
        """Return a string representing the node in g2o format
        """
        return "VERTEX_SE2 {} {} {} {}".format(self.id_, self.x, self.y,
                                               self.theta)


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

    def to_g2o(self):
        """Return a string representing the edge in g2o format
        """
        result = "EDGE_SE2 {} {} {} {} {} ".format(self.i, self.j, self.x, self.y,
                                               self.theta)
        result += " ".join([str(x) for x in self.info])
        return result


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
        return multi_graph

class MultiRobotGraph:
    """Multi robot graph

    TODO(Jay) Currently assumes 2 robots in 2D, extend to many robots in 3D

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
        self.odoms = [[] for _ in range(self.N)]
        self.lc = [[] for _ in range(self.N)]
        self.inter_lc = []

    def read_from(self, fpath):
        """Read multi robot graph from g2o file

        Args:
            fpath: input g2o file path
        """
        pass # TODO

    def read_nodes(self, nodes):
        """Split single robot nodes into nodes for 2 robots
        """
        segment_len = len(nodes)//self.N
        for k, v in nodes.items():
            idx = k // segment_len
            assert(idx < self.N)
            self.nodes[idx][k] = v

    def read_edges(self, odom_edges, loop_closure_edges):
        """Split single robot edges into edges for 2 robots
        """
        for odom in odom_edges:
            for idx, nodes in enumerate(self.nodes):
                if odom.i in nodes or odom.j in nodes:
                    self.odoms[idx].append(odom)
                    break

        for lc in loop_closure_edges:
            is_self_lc = False
            for idx, nodes in enumerate(self.nodes):
                if lc.i in nodes and lc.j in nodes:
                    self.lc[idx].append(lc)
                    is_self_lc = True
                    break
            if not is_self_lc:
                self.inter_lc.append(lc)

    def write_to(self, fpath):
        """Write graph to file
        """
        fp = open(fpath, "w+")
        for nodes in self.nodes:
            for node in nodes.values():
                fp.write(node.to_g2o() + "\n")
        for edges in self.odoms + self.lc + [self.inter_lc]:
            for edge in edges:
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

    def add_random_inter_lc(self, N=90):
        """Add randomly generated inter loop closures
        """
        x_mu = random.uniform(-5, 5)
        x_sigma = random.uniform(-2, 2)
        y_mu = random.uniform(-5, 5)
        y_sigma = random.uniform(-2, 2)
        theta_mu = random.uniform(-math.pi, math.pi)
        theta_sigma = random.uniform(-0.5, 0.5)

        info = [1/x_sigma**2, 0, 0, 1/y_sigma**2, 0, 1/theta_sigma**2]

        random_inter_lc = [Edge(random.choice(list(self.nodes[0])),
                                random.choice(list(self.nodes[1])),
                                random.normalvariate(x_mu, x_sigma),
                                random.normalvariate(y_mu, y_sigma),
                                random.normalvariate(theta_mu, theta_sigma),
                                info)
                           for _ in range(N)]
        self.inter_lc += random_inter_lc

    def add_perceptual_aliasing_lc(self, M=2, N=5):
        """Add perceptual aliasing loop closures

        Args:
            M: number of groups of aliases
            N: number of loop closures in each group
        """
        pass # TODO(Jay) Implement this
