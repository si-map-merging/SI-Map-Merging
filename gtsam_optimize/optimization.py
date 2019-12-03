import gtsam
import numpy as np

def vector3(x, y, z):
    """Create 3d double numpy array."""
    return np.array([x, y, z], dtype=np.float)


class Graph:
    """GTSAM graph

    Attributes:
        graph: gtsam graph
        initial: initial estimates
        result: optimized results
        marginals: node marginals
    """
    def __init__(self, single_robot_graph):
        """Build GTSAM graph from single robot graph from process_g2o package
        """
        gtsam_graph = gtsam.NonlinearFactorGraph()
        # create a short-hand
        srg = single_robot_graph

        # Set up first pose as origin
        PRIOR_NOISE = gtsam.noiseModel_Diagonal.Sigmas(vector3(0.1, 0.1, 0.1))
        min_node_idx = min(srg.nodes)
        min_node = srg.nodes[min_node_idx]
        gtsam_graph.add(gtsam.PriorFactorPose2(min_node.id_,
                        gtsam.Pose2(*min_node.pose()),
                        PRIOR_NOISE))

        # Add odometry factors & loop closure factors
        for edge in srg.odom_edges.values() + srg.loop_closure_edges.values():
            i, j = edge.i, edge.j
            assert(edge.has_diagonal_info())
            noise = gtsam.noiseModel_Diagonal.Sigmas(
                            vector3(*edge.diagonal_sigmas()))
            gtsam_graph.add(gtsam.BetweenFactorPose2(
                             i, j, gtsam.Pose2(*edge.measurement()), noise))

        # Create initial estimates
        initial_estimates = gtsam.Values()
        for node in srg.nodes.values():
            initial_estimates.insert(node.id_, gtsam.Pose2(*node.pose()))

        self.graph = gtsam_graph
        self.initial = initial_estimates

    def optimize(self):
        """Optimize the graph
        """
        gtsam_graph = self.graph
        initial_estimates = self.initial

        # Optimize
        parameters = gtsam.GaussNewtonParams()
        parameters.setRelativeErrorTol(1e-5)
        parameters.setMaxIterations(100)
        optimizer = gtsam.GaussNewtonOptimizer(gtsam_graph, initial_estimates,
                                               parameters)
        result = optimizer.optimize()

        self.result = result
        self.marginals = gtsam.Marginals(self.graph, self.result)

    def print_stats(self):
        """Print statistics of gtsam optimization
        """
        gtsam_graph = self.graph
        print("initial error = {}".format(gtsam_graph.error(self.initial)))
        print("final error = {}".format(gtsam_graph.error(self.result)))

    def get_covariance(self, idx):
        return self.marginals.marginalCovariance(idx)

    def write_to(self, fpath):
        """Write the optimized graph as g2o file
        """
        gtsam.writeG2o(self.graph, self.result, fpath)

