"""
Build the adjacency matrix
"""
import argparse
from scipy import io, sparse
import numpy as np
from tqdm import tqdm
from process_g2o.utils import SingleRobotGraph, Edge
from gtsam_optimize.optimization import Graph


class AdjacencyMatrix:
    """
    The major class for building the adjacency matrix
    """
    def __init__(self, multi_rob_graph, gamma=3, optim=True):
        self.optim = optim
        self.gamma = gamma
        self.graph = multi_rob_graph
        self.inter_lc_n = len(multi_rob_graph.inter_lc)
        self.inter_lc_edges = list(multi_rob_graph.inter_lc.values())
        graph1, graph2 = self.graph.to_singles()
        self.gtsam_graph1 = Graph(graph1)
        self.gtsam_graph2 = Graph(graph2)

    def single_graphs_optimization(self):
        """
        Optimize the single robot graphs using gtsam, see optimization.py
        """
        print("=========== Single Graphs Optimization ==============")
        self.gtsam_graph1.optimize()
        self.gtsam_graph2.optimize()
        self.gtsam_graph1.print_stats()
        self.gtsam_graph2.print_stats()

    def get_trusted_lc(self, indices):
        """Get trusted loop closure edges

        Args:
            indices: list of trusted indices, indexed from 1
        """
        trusted = []
        for idx in indices:
            trusted.append(self.inter_lc_edges[idx-1])
        return trusted

    def build_adjacency_matrix(self):
        """
        The central function in AdjacencyMatrix class.
        Return: A symmetric matrix whose entries are either 0 or 1
        """
        adjacency_matrix = np.zeros((self.inter_lc_n, self.inter_lc_n))
        for i in tqdm(range(self.inter_lc_n)):
            adjacency_matrix[i, i] = 1
            for j in tqdm(range(i)):
                mahlij = self.compute_mahalanobis_distance(self.inter_lc_edges[i], \
                         self.inter_lc_edges[j])
                print("this mahlij for {} is: {}".format((i+1, j+1), mahlij))
                if (mahlij <= self.gamma):
                    mahlji = self.compute_mahalanobis_distance(self.inter_lc_edges[j], \
                                                                self.inter_lc_edges[i])
                    print("this mahlij for {} is: {}".format((j+1, i+1), mahlji))
                    if mahlji <= self.gamma:
                        adjacency_matrix[j, i] = 1
                        adjacency_matrix[i, j] = 1

        assert self.check_symmetry(adjacency_matrix)
        print('The size of adjacency matrix is: ')
        print(adjacency_matrix.shape)
        sparse_adj_matrix = sparse.csr_matrix(adjacency_matrix)
        coo_adj_matrix = sparse_adj_matrix.tocoo()
        return coo_adj_matrix

    @classmethod
    def check_symmetry(cls, adj_matrix):
        """Check if the adjacency matrix is symmetric"""
        return np.allclose(adj_matrix, np.transpose(adj_matrix))

    def compute_mahalanobis_distance(self, edge1, edge2):
        """
        Input: edge1: Edge object
               edge2: Edge object
        """
        z_ik = edge1
        z_jl = edge2
        ii = z_ik.i
        kk = z_ik.j
        jj = z_jl.i
        ll = z_jl.j
        if not self.optim:
            x_ij = self.compute_current_estimate(ii, jj, 'a')
            x_lk = self.compute_current_estimate(ll, kk, 'b')
        else:
            x_ij = self.compute_current_estimate_after_optimization(ii, jj, 'a')
            x_lk = self.compute_current_estimate_after_optimization(ll, kk, 'b')
        new_edge = self.compound_op(self.compound_op(self.compound_op( \
                                    self.inverse_op(z_ik), x_ij), z_jl), x_lk)
        s = np.array([[new_edge.x, new_edge.y, new_edge.theta]])
        sigma = self.get_covariance(new_edge)
        return np.matmul(np.matmul(s, np.linalg.inv(sigma)), s.T)[0][0]

    def compute_current_estimate(self, start, end, robot_idx):
        """
        Compute intra-robot pose and return an Edge object
        Input: Start index and end index of robot_idx
        Output: An Edge object
        """
        isreversed = start > end

        if robot_idx == 'a':
            odoms = self.graph.odoms[0]
        else:
            odoms = self.graph.odoms[1]

        if not isreversed:
            trans_pose = odoms[(start, start+1)]

            for i in range(start+1, end):
                next_edge = odoms[(i, i+1)]
                trans_pose = self.compound_op(trans_pose, next_edge)
        else:
            start, end = end, start
            trans_pose = odoms[(start, start+1)]
            for j in range(start+1, end):
                next_edge = odoms[(j, j+1)]
                trans_pose = self.compound_op(trans_pose, next_edge)
            trans_pose = self.inverse_op(trans_pose)
        return trans_pose

    def compute_current_estimate_after_optimization(self, start, end, robot_idx):
        """
        Using the gtsam optimzation info to compute the current estimation
        Input: Start `node` index and end `node` index of robot_idx
        Output: An Edge object
        """
        start_pose = self.optimized_node_to_virtual_edge(start, robot_idx)
        end_pose = self.optimized_node_to_virtual_edge(end, robot_idx)
        trans_pose = self.inverse_compound(start_pose, end_pose, robot_idx)

        return trans_pose


    def optimized_node_to_virtual_edge(self, idx, robot_idx):
        """
        Convert a post-optimization Node with covariance to a 'virtual Edge'. The
        first index is 'w', meaning world. We are estimating from the world frame
        to that node. The reason doing this is to make it easy to use the inverse_op
        and compound_op operations to get new Edge objects.

        Input: the index of the pose: idx
               the index of the robot: robot_idx
        Output: an Edge object
        """
        if robot_idx == 'a':
            pose = self.gtsam_graph1.get_pose(idx)
            cov = self.gtsam_graph1.cov(idx)
            info = self.to_info(cov)
        elif robot_idx == 'b':
            pose = self.gtsam_graph2.get_pose(idx)
            cov = self.gtsam_graph2.cov(idx)
            info = self.to_info(cov)
        return Edge('w', idx, pose[0], pose[1], pose[2], info)

    def inverse_op(self, pose):
        """
        Compute x_ji given x_ij
        Input: An Edge object
        Output: An Edge object
        """
        x = pose.x
        y = pose.y
        theta = pose.theta
        cov = self.get_covariance(pose)

        new_x = -x*np.cos(theta)-y*np.sin(theta)
        new_y = x*np.sin(theta)-y*np.cos(theta)
        new_theta = -theta

        J_minus = np.matrix([[-np.cos(theta), -np.sin(theta), new_y], \
                             [np.sin(theta), -np.cos(theta), -new_x], \
                             [0, 0, -1]])
        new_cov = np.matmul(np.matmul(J_minus, cov), J_minus.T)
        new_info = self.to_info(new_cov)
        return Edge(pose.j, pose.i, new_x, new_y, new_theta, new_info)

    def compound_op(self, pose1, pose2):
        """
        Compute pose1 circle+ pose2
        Input: Two Edge objects pose1 and pose2
        Output: An Edge object
        """
        x1, y1, theta1 = pose1.x, pose1.y, pose1.theta
        x2, y2, theta2 = pose2.x, pose2.y, pose2.theta
        new_x = x2*np.cos(theta1) - y2*np.sin(theta1) + x1
        new_y = x2*np.sin(theta1) + y2*np.cos(theta1) + y1
        new_theta = theta1 + theta2
        cov1 = self.get_covariance(pose1)
        cov2 = self.get_covariance(pose2)
        cross_cov = self.get_cross_covariance()

        # prev_cov is a 6x6 matrix
        prev_cov = np.zeros((6, 6))
        prev_cov[0:3, 0:3] = cov1
        prev_cov[0:3, 3:6] = cross_cov
        prev_cov[3:6, 0:3] = cross_cov.T
        prev_cov[3:6, 3:6] = cov2

        J_plus = np.matrix([[1, 0, -(new_y-y1), np.cos(theta1), -np.sin(theta1), 0], \
                            [0, 1, (new_x-x1), np.sin(theta1), np.cos(theta1), 0], \
                            [0, 0, 1, 0, 0, 1]])

        new_cov = np.matmul(np.matmul(J_plus, prev_cov), J_plus.T)
        new_info = self.to_info(new_cov)
        return Edge(pose1.i, pose2.j, new_x, new_y, new_theta, new_info)

    def inverse_compound(self, pose1, pose2, robot_idx):
        """
        Compounding two optimized robot poses (Node), by considering covariance and
        cross covariance of the two poses to be compounded.
        Input: Two virtual Edge objects, and the robot index 'a' or 'b'
        Output: An Edge object
        """
        pose1_inversed = self.inverse_op(pose1)
        pose1, pose1_orig = pose1_inversed, pose1
        if robot_idx == 'a':
            single_graph = self.gtsam_graph1
        elif robot_idx == 'b':
            single_graph = self.gtsam_graph2
        theta1_o = pose1_orig.theta
        x1, y1, theta1 = pose1.x, pose1.y, pose1.theta
        x2, y2, theta2 = pose2.x, pose2.y, pose2.theta
        new_x = x2*np.cos(theta1) - y2*np.sin(theta1) + x1
        new_y = x2*np.sin(theta1) + y2*np.cos(theta1) + y1
        new_theta = theta1 + theta2
        assert pose1.j == pose2.i == 'w'
        cov1 = single_graph.cov(pose1.i)
        cov2 = single_graph.cov(pose2.j)
        inversed_cross_cov = single_graph.cross_cov(pose1.i, pose2.j)
        J_minus = np.matrix([[-np.cos(theta1_o), -np.sin(theta1_o), y1], \
                             [np.sin(theta1_o), -np.cos(theta1_o), -x1], \
                             [0, 0, -1]])
        cross_cov = np.matmul(J_minus, inversed_cross_cov)
        # prev_cov is a 6x6 matrix
        prev_cov = np.zeros((6, 6))
        prev_cov[0:3, 0:3] = cov1
        prev_cov[0:3, 3:6] = cross_cov
        prev_cov[3:6, 0:3] = cross_cov.T
        prev_cov[3:6, 3:6] = cov2
        J_plus = np.matrix([[1, 0, -(new_y-y1), np.cos(theta1), -np.sin(theta1), 0], \
                            [0, 1, (new_x-x1), np.sin(theta1), np.cos(theta1), 0], \
                            [0, 0, 1, 0, 0, 1]])
        new_cov = np.matmul(np.matmul(J_plus, prev_cov), J_plus.T)
        new_info = self.to_info(new_cov)
        return Edge(pose1.i, pose2.j, new_x, new_y, new_theta, new_info)

    def get_covariance(self, pose):
        """
        Get the covariance matrix given an Edge object
        Input: An Edge object
        Output: A numpy array
        """
        info = np.zeros((3, 3))
        info[0, 0:3] = pose.info[0:3]
        info[1, 1:3] = pose.info[3:5]
        info[2, 2] = pose.info[5]
        info_mat = info + info.T - np.diag(info.diagonal())
        cov_mat = np.linalg.inv(info_mat)
        assert self.check_symmetry(cov_mat)
        return cov_mat

    @classmethod
    def get_cross_covariance(cls):
        """
        Compute the cross covariance of pose1 and pose2
        Note: Currently assumming a zero matrix, meaning we assume the measurements
        are independent to each other
        Input: Two Edge objects pose1 and pose2
        Output: A numpy matrix
        """
        return np.zeros((3, 3))

    @classmethod
    def to_info(cls, cov):
        """
        Convert the covariance matrix to info (6x1 vector in 2D)
        Input: A covariance matrix
        Output: A vector
        """
        info_mat = np.linalg.inv(cov)
        info = [info_mat[0, 0], info_mat[0, 1], info_mat[0, 2], \
                info_mat[1, 1], info_mat[1, 2], info_mat[2, 2]]
        return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the adjacency matrix given one g2o file")
    parser.add_argument("input_fpath", metavar="city10000.g2o", type=str, nargs='?',
                        default="datasets/manhattanOlson3500.g2o", help="g2o file path")
    parser.add_argument("output_fpath", metavar="adjacency.mtx", type=str, nargs='?',
                        default="adjacency.mtx", help="adjacency file path")
    args = parser.parse_args()

    graph = SingleRobotGraph()
    graph.read_from(args.input_fpath)
    print("========== Input g2o Graph Summary ================")
    graph.print_summary()

    multi_graph = graph.to_multi()
    multi_graph.add_random_inter_lc()
    print("========== Multi Robot g2o Graph Summary ================")
    multi_graph.print_summary()

    adj = AdjacencyMatrix(multi_graph, gamma=0.1, optim=True)
    adj.single_graphs_optimization()
    coo_adj_mat = adj.build_adjacency_matrix()
    io.mmwrite(args.output_fpath, coo_adj_mat, symmetry='symmetric')
