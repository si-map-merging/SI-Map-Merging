"""
GTSAM Optimization
"""
import numpy as np
import gtsam



def vector3(x, y, z):
    """Create 3d double numpy array."""
    return np.array([x, y, z], dtype=np.float)


def vector6(x, y, z, d, e, f):
    """Create 3d double numpy array."""
    return np.array([x, y, z, d, e, f], dtype=np.float)


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
        self.graph = gtsam.NonlinearFactorGraph()
        self.srg = single_robot_graph

        self.set_anchor()
        self.add_factors()
        self.create_initial()

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

    def cov(self, idx):
        return self.marginals.marginalCovariance(idx)

    def cross_cov(self, i, j):
        key_vec = gtsam.gtsam.KeyVector()
        key_vec.push_back(i)
        key_vec.push_back(j)
        return self.marginals.jointMarginalCovariance(key_vec).at(i, j)

    def joint_marginal(self, i, j):
        key_vec = gtsam.gtsam.KeyVector()
        key_vec.push_back(i)
        key_vec.push_back(j)
        return self.marginals.jointMarginalCovariance(key_vec).fullMatrix()

    def get_pose(self, idx):
        """After optimization, extract the pose of index idx

        Return:
            pose as np.array([x, y, theta])
        """
        pose = self.result.atPose2(idx)
        return np.array([pose.x(), pose.y(), pose.theta()])

    def write_to(self, fpath):
        """Write the optimized graph as g2o file
        """
        gtsam.writeG2o(self.graph, self.result, fpath)


class Graph2D(Graph):
    def set_anchor(self):
        """Set prior on the first node, to make it an anchor
        """
        srg = self.srg
        gtsam_graph = self.graph

        # Set up first pose as origin
        PRIOR_NOISE = gtsam.noiseModel_Diagonal.Sigmas(vector3(0.1, 0.1, 0.1))
        min_node_idx = min(srg.nodes)
        min_node = srg.nodes[min_node_idx]
        gtsam_graph.add(gtsam.PriorFactorPose2(min_node.id_,
                        gtsam.Pose2(*min_node.pose()),
                        PRIOR_NOISE))

    def add_factors(self):
        """Add odometry factors & loop closure factors
        """
        srg = self.srg
        gtsam_graph = self.graph
        for edge in list(srg.odom_edges.values()) + list(
                     srg.loop_closure_edges.values()):
            i, j = edge.i, edge.j
            assert(edge.has_diagonal_info())
            noise = gtsam.noiseModel_Diagonal.Sigmas(
                            vector3(*edge.diagonal_sigmas()))
            gtsam_graph.add(gtsam.BetweenFactorPose2(
                             i, j, gtsam.Pose2(*edge.measurement()), noise))

    def create_initial(self):
        """Create initial estimates
        """
        initial_estimates = gtsam.Values()
        for node in self.srg.nodes.values():
            initial_estimates.insert(node.id_, gtsam.Pose2(*node.pose()))
        self.initial = initial_estimates


class Graph3D(Graph):
    def set_anchor(self):
        """Set prior on the first node, to make it an anchor
        """
        srg = self.srg
        gtsam_graph = self.graph

        # Set up first pose as origin
        PRIOR_NOISE = gtsam.noiseModel_Diagonal.Variances(
            vector6(1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4))
        min_node_idx = min(srg.nodes)
        min_node = srg.nodes[min_node_idx]
        gtsam_graph.add(gtsam.PriorFactorPose3(min_node.id_,
                        gtsam.Pose3(min_node.pose()),
                        PRIOR_NOISE))

    def add_factors(self):
        """Add odometry factors & loop closure factors
        """
        srg = self.srg
        gtsam_graph = self.graph
        for edge in list(srg.odom_edges.values()) + list(
                     srg.loop_closure_edges.values()):
            i, j = edge.i, edge.j
            noise = gtsam.noiseModel_Gaussian.Covariance(edge.cov())
            gtsam_graph.add(gtsam.BetweenFactorPose3(
                             i, j, gtsam.Pose3(edge.measurement()), noise))

    def create_initial(self):
        """Create initial estimates
        """
        initial_estimates = gtsam.Values()
        for node in self.srg.nodes.values():
            initial_estimates.insert(node.id_, gtsam.Pose3(node.pose()))
        self.initial = initial_estimates

    def get_pose(self, idx):
        """After optimization, extract the pose of index idx

        Return:
            pose as [translation quaternion]
        """
        pose = self.result.atPose3(idx)
        return [pose.translation().vector(), pose.rotation().quaternion()]

if __name__ == "__main__":
    import sys
    sys.path.append("../")
    from process_g2o.utils import SingleRobotGraph3D, SingleRobotGraph2D

    is_3D = True
    if is_3D:
        srg = SingleRobotGraph3D()
        srg.read_from("datasets/parking-garage.g2o")
        gtsam_graph = Graph3D(srg)
    else:
        srg = SingleRobotGraph2D()
        srg.read_from("../datasets/manhattanOlson3500.g2o")
        gtsam_graph = Graph2D(srg)


    gtsam_graph.optimize()
    print("======= Manhattan Graph Optimization =========")
    gtsam_graph.print_stats()
    pose = gtsam_graph.get_pose(39)
    i = 0
    print("======= Covariance of Node {} ===========".format(i))
    print(gtsam_graph.cov(i))

    j = 100
    print("===== Cross Covariance between Node {} and {} ======".format(i, j))
    print(gtsam_graph.cross_cov(i, j))

    print("====== Joint Marginal of Node {} and {} =======".format(i, j))
    print(gtsam_graph.joint_marginal(i, j))
