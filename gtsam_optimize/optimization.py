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
        pose2 = self.result.atPose2(idx)
        return np.array([pose2.x(), pose2.y(), pose2.theta()])

    def between(self, idxi, idxj):
        """Compute the compounded 2D pose between pose i and pose j, and the resulting
        covariance between pose i and pose j
        Return:
        result: a 3x1 numpy array
                the covariance between pose i and pose j
        """
        result, gtsam_cov = self.pos_and_cov(idxi, idxj)

        result_pose = np.array([result.x(), result.y(), result.theta()])
        return result_pose, gtsam_cov.covariance()

    def pos_and_cov(self, idxi, idxj):
        """Return relative Pose2D between pose i and pose j, and the covariance
        """
        posei = self.result.atPose2(idxi)
        posej = self.result.atPose2(idxj)
        result = posei.inverse().compose(posej)
        H1 = -result.inverse().AdjointMap()
        size = H1.shape[0]
        H2 = np.eye(size)
        A = np.hstack([H1, H2])
        covij = self.joint_marginal(idxi, idxj)

        cov = A @ covij @ A.T
        return result, gtsam.noiseModel_Gaussian.Covariance(cov)

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

    def add_SI_factors_odometry(self,b_odometry_edges_list):
        '''
        add scale invariane between factor
        element in b_odometry_edges_list are consequent odometry poses of robot b
        for example: [(edge1, edge2), (edge2,edge3),...]
        edge1.i = 1
        edge1.j = 2
        edge2.i = 2
        edge2.j = 3
        ...
        '''
        gtsam_graph = self.graph
        for edge1, edge2 in b_odometry_edges_list:
            i, j ,k= edge1.i, edge1.j, edge2.j
            xij = gtsam.Pose2(edge1.x,edge1.y,edge1.theta)
            xjk = gtsam.Pose2(edge2.x,edge2.y,edge2.theta)

            noiseij = gtsam.noiseModel_Diagonal.Sigmas(
                            vector3(*edge1.diagonal_sigmas())).covariance()
            noiseik = gtsam.noiseModel_Diagonal.Sigmas(
                            vector3(*edge2.diagonal_sigmas())).covariance()

            xik, noiseik = compound_pose(xij,xjk,noiseij,noiseik)

            measured,noise,H=construct_SIFactor2(xij,noiseij)
            gtsam_graph.add(gtsam.SIBetweenFactorPose2(i, j, measured, noise, H))

            measured,noise,H=construct_SIFactor2(xjk,noisejk)
            gtsam_graph.add(gtsam.SIBetweenFactorPose2(j, k, measured, noise, H))

            measured,noise,H=construct_SIFactor2(xik,noiseik)
            gtsam_graph.add(gtsam.SIBetweenFactorPose2(i, k, measured, noise, H))

    def add_lc_factor(self, lc_list):
        '''
        lc_list is loop colsure between a,b
        '''
        gtsam_graph = self.graph

        for edge in lc_list:
            i, j = edge.i, edge.j
            noise = gtsam.noiseModel_Diagonal.Sigmas(
                            vector3(*edge.diagonal_sigmas())).covariance()
            x = gtsam.Pose2(edge.x,edge.y,edge.theta)

            measured,noise,H=construct_SIFactor2(x,noise)
            gtsam_graph.add(gtsam.SIBetweenFactorPose2(i, j, measured, noise, H))


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
        return [pose.translation().vector(), pose.rotation().matrix()]

    def between(self, idxi, idxj):
        """Compute the compounded 2D pose between pose i and pose j, and the resulting
        covariance between pose i and pose j
        Return:
        result: a 3
        """
        posei = self.result.atPose3(idxi)
        posej = self.result.atPose3(idxj)
        result = posei.inverse().compose(posej)
        result_pose = [result.translation().vector(), result.rotation().matrix()]
        H1 = -result.inverse().AdjointMap()
        size = H1.shape[0]
        H2 = np.eye(size)
        A = np.hstack([H1, H2])
        covi = self.cov(idxi)
        covj = self.cov(idxj)
        cross_cov = self.cross_cov(idxi, idxj)
        covij = np.zeros((12, 12))
        covij[:6, :6] = covi
        covij[:6, 6:] = cross_cov
        covij[6:, :6] = cross_cov.T
        covij[6:, 6:] = covj

        return result_pose, A @ covij @ A.T

if __name__ == "__main__":
    import sys
    sys.path.append("../")
    from process_g2o.utils import SingleRobotGraph3D, SingleRobotGraph2D

    is_3D = False
    if is_3D:
        srg = SingleRobotGraph3D()
        srg.read_from("../datasets/parking-garage.g2o")
        graph = Graph3D(srg)
    else:
        srg = SingleRobotGraph2D()
        srg.read_from("../datasets/manhattanOlson3500.g2o")
        graph = Graph2D(srg)


    graph.optimize()
    print("======= Manhattan Graph Optimization =========")
    graph.print_stats()
    pose = graph.get_pose(39)
    i = 0
    print("======= Covariance of Node {} ===========".format(i))
    print(graph.cov(i))

    j = 100
    print("===== Cross Covariance between Node {} and {} ======".format(i, j))
    print(graph.cross_cov(i, j))

    print(graph.cross_cov(j, i))

    print("====== Joint Marginal of Node {} and {} =======".format(i, j))
    print(graph.joint_marginal(i, j))
