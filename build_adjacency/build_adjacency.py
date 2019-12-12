"""
Build the adjacency matrix
"""
import argparse
import itertools
from scipy import io, sparse
import numpy as np
from tqdm import tqdm
import sophus as sp
from process_g2o.utils import MultiRobotGraph2D, Edge2D, MultiRobotGraph3D, \
    Edge3D, Quaternion, cholesky_inverse
from gtsam_optimize.optimization import Graph2D, Graph3D
import gtsam
from scale_estimation.scale_estimate import ScaleEstimation


class AdjacencyMatrix:
    """
    The major class for building the adjacency matrix
    """
    def __init__(self, multi_rob_graph, gamma=3, optim=True, dim2=True):
        """
        Args:
            multi_rob_graph: multi-robot graph of type MultiRobotGraph
            gamma: threshold for checking adjacency
            optim: whether perform single-robot optimization as a
                preprocessing step
        """
        self.optim = optim
        self.gamma = gamma
        self.graph = multi_rob_graph
        self.inter_lc_n = len(multi_rob_graph.inter_lc)
        self.inter_lc_edges = list(multi_rob_graph.inter_lc.values())
        self.scale_estimator = ScaleEstimation(self.inter_lc_n)
        self.gtsam_graph1 = None
        self.gtsam_graph2 = None
        if optim and dim2:
            self.build_gtsam_graphs()

    def build_gtsam_graphs(self):
        graph1, graph2 = self.graph.to_singles()
        self.gtsam_graph1 = Graph2D(graph1)
        self.gtsam_graph2 = Graph2D(graph2)
        # Single graphs optimization
        print("=========== 2D Single Graphs Optimization ==============")
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
        """The central function in AdjacencyMatrix class.

        Return: A symmetric matrix whose entries are either 0 or 1
        """
        self.feed_lc()
        self.correct_for_scale()
        self.build_gtsam_graphs()

        adjacency_matrix = np.zeros((self.inter_lc_n, self.inter_lc_n))
        for i in tqdm(range(self.inter_lc_n)):
            adjacency_matrix[i, i] = 1
            for j in tqdm(range(i)):
                mahlij = self.compute_mahalanobis_distance(self.inter_lc_edges[i], \
                         self.inter_lc_edges[j])
                # print("this mahlij for {} is: {}".format((i+1, j+1), mahlij))
                if (mahlij <= self.gamma):
                    mahlji = self.compute_mahalanobis_distance(self.inter_lc_edges[j], \
                                                                self.inter_lc_edges[i])
                    # print("this mahlji for {} is: {}".format((j+1, i+1), mahlji))
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
        Args: edge1: Edge object
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
            x_ij = self.get_current_estimate_from_gtsam(ii, jj, 'a')
            x_lk = self.get_current_estimate_from_gtsam(ll, kk, 'b')
        new_edge = self.compound_op(self.compound_op(self.compound_op( \
                                    self.inverse_op(z_ik), x_ij), z_jl), x_lk)
        s = np.array([[new_edge.x, new_edge.y, new_edge.theta]])
        info_mat = self.get_info_mat(new_edge)
        return np.matmul(np.matmul(s, info_mat), s.T)[0][0]

    def compute_current_estimate(self, start, end, robot_idx):
        """Compute intra-robot pose and return an Edge object

        Args:
            start: start index
            end: end index
            robot_idx: robot index
        Return:
            An computed edge from start to end as an Edge object
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
        """Using the gtsam optimzation info to compute the current estimation

        Args:
            start: start index
            end: end index
            robot_idx: robot index
        Return:
            An computed edge from start to end as an Edge object
        """
        start_pose = self.optimized_node_to_virtual_edge(start, robot_idx)
        end_pose = self.optimized_node_to_virtual_edge(end, robot_idx)
        assert end_pose.i != end_pose.j
        trans_pose = self.inverse_compound(start_pose, end_pose, robot_idx)

        return trans_pose

    def get_current_estimate_from_gtsam(self, start, end, robot_idx):
        """Using gtsam's between function to directly get current estimation
        Return:
            An Edge2D object describing the transformation from start to end
        """
        if robot_idx == 'a':
            gtsam_graph = self.gtsam_graph1
        elif robot_idx == 'b':
            gtsam_graph = self.gtsam_graph2

        transform, covariance = gtsam_graph.between(start, end)
        info = self.to_info(covariance)
        return Edge2D(start, end, transform[0], transform[1], transform[2], info)

    def optimized_node_to_virtual_edge(self, idx, robot_idx):
        """Convert a post-optimization Node with covariance to a 'virtual Edge'. The
        first index is 'w', meaning world. We are estimating from the world frame
        to that node. The reason doing this is to make it easy to use the inverse_op
        and compound_op operations to get new Edge objects.

        Args:
            idx: the index of the pose
            robot_idx: the index of the robot
        Return:
            an Edge object
        """
        if robot_idx == 'a':
            pose = self.gtsam_graph1.get_pose(idx)
            cov = self.gtsam_graph1.cov(idx)
            info = self.to_info(cov)
        elif robot_idx == 'b':
            pose = self.gtsam_graph2.get_pose(idx)
            cov = self.gtsam_graph2.cov(idx)
            info = self.to_info(cov)
        return Edge2D('w', idx, pose[0], pose[1], pose[2], info)

    def inverse_op(self, pose):
        """Compute x_ji given x_ij

        Args:
            pose: an Edge object representing the edge from world to pose
        Return:
            the inversed Edge object
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
        return Edge2D(pose.j, pose.i, new_x, new_y, new_theta, new_info)

    def compound_op(self, pose1, pose2):
        """Compute pose1 circle+ pose2

        Args:
            Two Edge objects pose1 and pose2
        Return:
            An Edge object
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
        return Edge2D(pose1.i, pose2.j, new_x, new_y, new_theta, new_info)

    def inverse_compound(self, pose1, pose2, robot_idx):
        """Compounding two optimized robot poses (Node), by considering covariance and
        cross covariance of the two poses to be compounded.

        Args:
            Two virtual Edge objects, and the robot index 'a' or 'b'
        Return:
            An Edge object
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
        cov1 = self.get_covariance(pose1)  # The inversed pose1's covariance
        cov2 = self.get_covariance(pose2)
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
        return Edge2D(pose1.i, pose2.j, new_x, new_y, new_theta, new_info)

    @classmethod
    def is_pos_def(cls, matrix):
        """Check if a matrix is positive definite
        Computing cholesky decomposition is much more efficient than computing eigenvalues.
        Output: True or False
        """
        try:
            np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError:
            return False
        return True

    def get_info_mat(self, pose):
        """Extract information matrix from pose
        """
        info = np.zeros((3, 3), dtype=np.float)
        info[0, 0:3] = pose.info[0:3]
        info[1, 1:3] = pose.info[3:5]
        info[2, 2] = pose.info[5]
        info_mat = info + info.T - np.diag(info.diagonal())
        assert self.check_symmetry(info_mat)
        return info_mat

    def get_covariance(self, pose):
        """Get the covariance matrix given an Edge object

        Args:
            An Edge object
        Return:
            A numpy array
        """
        info_mat = self.get_info_mat(pose)
        assert self.is_pos_def(info_mat), "info_mat is not positive definite!"
        # cov_mat = cholesky_inverse(info_mat)
        cov_mat = np.linalg.inv(info_mat)
        # try:
        #     cov_mat = cholesky_inverse(info_mat)
        # except np.linalg.LinAlgError:
        #     __import__("pdb").set_trace()

        assert self.check_symmetry(cov_mat)
        return cov_mat

    @classmethod
    def get_cross_covariance(cls):
        """Compute the cross covariance of pose1 and pose2
        Note: Currently assumming a zero matrix, meaning we assume the measurements
        are independent to each other

        Args:
            Two Edge objects pose1 and pose2
        Return:
            A numpy matrix
        """
        return np.zeros((3, 3))

    @classmethod
    def to_info(cls, cov):
        """Convert the covariance matrix to info (6x1 vector in 2D)

        Args:
            A covariance matrix
        Return:
            A vector
        """
        info_mat = cholesky_inverse(cov)
        info = [info_mat[0, 0], info_mat[0, 1], info_mat[0, 2], \
                info_mat[1, 1], info_mat[1, 2], info_mat[2, 2]]
        return info

    def feed_lc(self):
        """Feed all combinations of 3 inter-robot loop closures into
            scale estimation module
        """
        # return
        combs = list(itertools.combinations(range(len(self.inter_lc_edges)), 3))
        for indices in tqdm(combs):
            i, j, k = indices
            edge1 = self.inter_lc_edges[i]
            edge2 = self.inter_lc_edges[j]
            edge3 = self.inter_lc_edges[k]

            def contain_same_nodes(edge1, edge2, edge3):
                indices = set([edge1.i, edge1.j, edge2.i, edge2.j, edge3.i,
                                edge3.j] )
                if len(indices) < 6:
                    return True
                return False

            if contain_same_nodes(edge1, edge2, edge3):
                continue

            # Fill loop closure values and covariances
            z_values = []
            Q_z_values = []
            for idx in indices:
                edge = self.inter_lc_edges[idx]
                z_values.append( gtsam.Pose2(*edge.measurement() ))
                Q_z_values.append( gtsam.noiseModel_Gaussian.Covariance(edge.cov()))

            # Compute relative poses
            x_a_ij, Q_a_ij = self.gtsam_graph1.pos_and_cov( edge1.i, edge2.i )
            x_a_jk, Q_a_jk = self.gtsam_graph1.pos_and_cov( edge2.i, edge3.i )
            x_a_ik, Q_a_ik = self.gtsam_graph1.pos_and_cov( edge1.i, edge3.i )

            x_b_ij, Q_b_ij = self.gtsam_graph2.pos_and_cov( edge1.j, edge2.j )
            x_b_jk, Q_b_jk = self.gtsam_graph2.pos_and_cov( edge2.j, edge3.j )
            x_b_ik, Q_b_ik = self.gtsam_graph2.pos_and_cov( edge1.j, edge3.j )

            # Assemble values
            poses = [[x_a_ij, x_a_jk, x_a_ik], [x_b_ij, x_b_jk, x_b_ik], z_values]
            covs = [[Q_a_ij, Q_a_jk, Q_a_ik], [Q_b_ij, Q_b_jk, Q_b_ik], Q_z_values]
            self.scale_estimator.scale_estimate(poses, covs, indices)


    def correct_for_scale(self):
        """Get the estimated scales, and correct the graph for these scales
        """
        #s_b, lc_scales = self.scale_estimator.get_scales()
        # s_b = 0.2
        # lc_scales = [1]*self.inter_lc_n
        s_b = self.scale_estimator.estimate_sb()
        lc_norms = self.scale_estimator.estimate_lc()

        # Correct for robot b scale
        self.graph.scale_robot_b( 1.0/s_b )

        # Scale inter-robot lc
        for i in range(self.inter_lc_n):
            s_l = lc_norms[i] / self.inter_lc_edges[i].norm()
            self.inter_lc_edges[i] *= s_l

        # Replace graph inter-robot lc with the corrected ones
        for lc in self.inter_lc_edges:
            key = (lc.i, lc.j)
            self.graph.inter_lc[key] = lc


class AdjacencyMatrix3D(AdjacencyMatrix):
    """Building adjacency matrix from single 3D pose graphs.
    """
    def __init__(self, multi_graph3D, gamma=0.1, optim=True):
        AdjacencyMatrix.__init__(self, multi_graph3D, optim=optim, dim2=False)
        if self.optim:
            graph1, graph2 = self.graph.to_singles()
            self.gtsam_graph1 = Graph3D(graph1)
            self.gtsam_graph2 = Graph3D(graph2)
            print("=========== 3D Single Graphs Optimization  ==============")
            self.gtsam_graph1.optimize()
            self.gtsam_graph2.optimize()
            self.gtsam_graph1.print_stats()
            self.gtsam_graph2.print_stats()

    def compute_mahalanobis_distance(self, edge1, edge2):
        """Compute Mahalanobis distance between two measurements
        Now edge1 and edge2 are Edge3D objects
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
            if kk != ll and ii != jj:
                x_ij = self.get_current_estimate_from_gtsam(ii, jj, 'a')
                x_lk = self.get_current_estimate_from_gtsam(ll, kk, 'b')
            else:               # two loop closures coincide
                if kk == ll:
                    x_ij = self.get_current_estimate_from_gtsam(ii, jj, 'a')
                elif ii == jj:
                    x_lk = self.get_current_estimate_from_gtsam(ll, kk, 'b')

        # for debug
        try:
            if kk != ll and ii != jj:
                new_edge = self.compound_op(self.compound_op(self.compound_op( \
                                            self.inverse_op(z_ik), x_ij), z_jl), x_lk)
            else:
                if kk == ll:
                    new_edge = self.compound_op(self.compound_op( \
                        self.inverse_op(z_ik), x_ij), z_jl)
                elif ii == jj:
                    new_edge = self.compound_op(self.compound_op( \
                        self.inverse_op(z_ik), z_jl), x_lk)

        except AssertionError:
            print('z_ik index: ' + str(z_ik.i) + ' ' + str(z_ik.j))
            print('z_jl index: ' + str(z_jl.i) + ' ' + str(z_jl.j))
        # try:
        #     inv = self.inverse_op(z_ik)
        #     new_edge1 = self.compound_op(inv, x_ij)
        #     new_edge2 = self.compound_op(new_edge1, z_jl)
        #     new_edge3 = self.compound_op(new_edge2, x_lk)
        #     new_edge = new_edge3
        # except AssertionError:
        #     __import__("pdb").set_trace()
        #     print('z_ik index: ' + str(z_ik.i) + ' ' + str(z_ik.j))
        #     print('z_jl index: ' + str(z_jl.i) + ' ' + str(z_jl.j))

        s = sp.SE3(new_edge.measurement()).log().flatten()  # # check this, heed sequence
        return np.matmul(np.matmul(s, new_edge.info_mat()), s.T)

    # def compute_current_estimate_after_optimization(self, start, end, robot_idx):
    #     """Using the gtsam optimization info to compute the current estimation
    #     of 3D Pose
    #     """
    #     return super().compute_current_estimate_after_optimization(start, end, robot_idx)

    def optimized_node_to_virtual_edge(self, idx, robot_idx):
        """Convert a Node3D object to (virtual) Edge3D object.
        Just the same as in the 2D case, setting the first index to 'w'.

        Args: the index of the pose: idx
               the index of the robot: robot_idx
        Return: an Edge3D object
        """
        if robot_idx == 'a':
            translation, R = self.gtsam_graph1.get_pose(idx)
            cov = self.gtsam_graph1.cov(idx)
            info = self.to_info(cov)
        elif robot_idx == 'b':
            translation, R = self.gtsam_graph2.get_pose(idx)
            cov = self.gtsam_graph2.cov(idx)
            info = self.to_info(cov)
        return Edge3D('w', idx, translation, Quaternion.from_R(R).q, info)

    def get_current_estimate_from_gtsam(self, start, end, robot_idx):
        """Using gtsam's between function to directly get current estimation
        Return:
            An Edge3D object describing the transformation from start to end
        """
        if robot_idx == 'a':
            gtsam_graph = self.gtsam_graph1
        elif robot_idx == 'b':
            gtsam_graph = self.gtsam_graph2

        transform, covariance = gtsam_graph.between(start, end)
        info = self.to_info(covariance)
        return Edge3D(start, end, transform[0], Quaternion.from_R(transform[1]).q, info)

    def inverse_op(self, pose):
        """Compute x_ji given x_ij in the 3D case.

        Args:
            Edge3D object
        Return:
            Edge3D object
        """
        T_inv = sp.SE3(pose.measurement()).inverse()
        R_inv = T_inv.rotationMatrix()
        q_inv = Quaternion.from_R(R_inv).q
        t_inv = T_inv.translation().flatten()  # [[],[],[]]->[,,]
        J_minus = self.compute_J_minus(pose)
        new_cov = np.matmul(np.matmul(J_minus, pose.cov()), J_minus.T)
        # try:
        #     assert self.check_symmetry(np.linalg.inv(new_cov))
        # except AssertionError:
        # __import__("pdb").set_trace()

        new_info = self.to_info(new_cov)
        return Edge3D(pose.j, pose.i, t_inv, q_inv, new_info)

    @classmethod
    def compute_J_minus(cls, pose):
        """Return the J_minus matrix for pose
        """
        T_inv = sp.SE3(pose.measurement()).inverse()
        # R_inv = T_inv.rotationMatrix()
        # q_inv = Quaternion.from_R(R_inv).q
        t_inv = T_inv.translation().flatten()  # [[],[],[]]->[,,]
        x_, y_, _ = t_inv
        R = pose.get_R()
        phi, theta, psi = pose.get_zyz()  # verify this
        x, y, z = pose.t
        n_x, n_y, n_z = R[:, 0]
        o_x, o_y, o_z = R[:, 1]
        a_x, a_y, a_z = R[:, 2]
        # x_ = -(n_x*x + n_y*y + n_z*z)
        # y_ = -(o_x*x + o_y*y + o_z*z)
        # z_ = -(a_x*x + a_y*y + a_z*z)
        # t_inv_manual = np.asarray([x_, y_, z_])
        # __import__("pdb").set_trace()
        # assert t_inv_manual.all() == t_inv.all()
        Q = np.matrix([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])
        N = np.matrix([[n_y*x-n_x*y, -n_z*x*np.cos(phi)-n_z*y*np.sin(phi)+z*np.cos(theta)*np.cos(psi), y_],
                       [o_y*x-o_x*y, -o_z*x*np.cos(phi)-o_z*y*np.sin(phi)-z*np.cos(theta)*np.sin(psi), -x_],
                       [a_y*x-a_x*y, -a_z*x*np.cos(phi)-a_z*y*np.sin(phi)+z*np.sin(theta), 0]])
        J_minus = np.zeros((6, 6))
        J_minus[0:3, 0:3] = -R.T
        J_minus[0:3, 3:] = N
        J_minus[3:, 3:] = Q
        return J_minus

    def compound_op(self, pose1, pose2, robot_idx=None, odom=False):
        """Compute pose1 circle+ pose2

        Args:
            Two Edge3D objects pose1 and pose2, measurement is a tag indifying
                whether the two poses are robot poses or measurements
        Return:
            an Edge3D object
        """
        assert pose1.j == pose2.i
        T1 = pose1.measurement()
        T2 = pose2.measurement()
        new_T = np.matmul(T1, T2)
        new_R = new_T[:3, :3]
        new_q = Quaternion.from_R(new_R).q
        new_t = new_T[:3, 3]

        R1 = pose1.get_R()
        # R2 = pose2.get_R()
        phi1, theta1, psi1 = pose1.get_zyz()
        _, theta2, psi2 = pose2.get_zyz()
        phi3, theta3, psi3 = Quaternion.to_zyz(new_q)
        x1, y1, z1 = pose1.t
        x2, y2, z2 = pose2.t
        x3, y3, z3 = new_t
        n_x1, n_y1, n_z1 = R1[:, 0]
        o_x1, o_y1, o_z1 = R1[:, 1]

        M = np.matrix([
            [-(y3-y1), (z3-z1)*np.cos(phi1), o_x1*x2-n_x1*y2],
            [x3-x1, (z3-z1)*np.sin(phi1), o_y1*x2-n_y1*y2],
            [0, -x2*np.cos(theta1)*np.cos(psi1)+y2*np.cos(theta1)*np.sin(psi1)-z2*np.sin(theta1), o_z1*x2-n_z1*y2]])

        K1 = np.matrix([
            [1, (np.cos(theta3)*np.sin(phi3-phi1))/np.sin(theta3), (np.sin(theta2)*np.cos(psi3-psi2))/np.sin(theta3)],
            [0, np.cos(phi3-phi1), np.sin(theta2)*np.sin(psi3-psi2)],
            [0, np.sin(phi3-phi1)/np.sin(theta3), (np.sin(theta1)*np.cos(phi3-phi1))/np.sin(theta3)]])

        K2 = np.matrix([
            [(np.sin(theta2)*np.cos(psi3-psi2))/np.sin(theta3), (np.sin(psi3-psi2))/np.sin(theta3), 0],
            [np.sin(theta2)*np.sin(psi3-psi2), np.cos(psi3-psi2), 0],
            [(np.sin(theta1)*np.cos(phi3-phi1))/np.sin(theta3), (np.cos(theta3)*np.sin(psi3-psi2))/np.sin(theta3), 1]])
        # J_plus is a 6x12 matrix
        J_plus = np.zeros((6, 12))
        J_plus[:3, :3] = np.identity(3)
        J_plus[:3, 3:6] = M
        J_plus[:3, 6:9] = R1
        J_plus[3:, 3:6] = K1
        J_plus[3:, 9:] = K2
        # prev_cov is a 12x12 matrix
        prev_cov = np.zeros((12, 12))
        cov1 = pose1.cov()
        cov2 = pose2.cov()
        cross_cov = self.get_cross_cov(pose1, pose2, robot_idx, odom)
        if odom:
            # J_minus_easy = self.compute_J_minus(pose1)
            # J_minus = np.linalg.inv(J_minus_easy)
            # print("J_minus_easy: \n")
            # print(J_minus_easy)
            # print("J_minus: \n")
            # print(J_minus)
            J_minus = self.compute_J_minus(self.inverse_op(pose1))
            # print("J_minus_hard: \n")
            # print(J_minus_hard)
            # print("Make sure change this later...")
            # assert np.allclose(J_minus, J_minus_hard)
            cross_cov = np.matmul(J_minus, cross_cov)
        prev_cov[:6, :6] = cov1
        prev_cov[:6, 6:] = cross_cov
        prev_cov[6:, :6] = cross_cov.T
        prev_cov[6:, 6:] = cov2

        new_cov = np.matmul(np.matmul(J_plus, prev_cov), J_plus.T)

        new_info = self.to_info(new_cov)
        return Edge3D(pose1.i, pose2.j, new_t, new_q, new_info)

    def get_cross_cov(self, pose1, pose2, robot_idx, odom):
        """The cross covariance matrix between two measurements. Assuming indenpendence
        when it comes to measurements, otherwise for odom, need to check the optimization
        results
        """
        if not odom:
            return np.zeros((6, 6))

        if robot_idx == 'a':
            single_graph = self.gtsam_graph1
        elif robot_idx == 'b':
            single_graph = self.gtsam_graph2
        assert pose1.j == pose2.i == 'w'
        inversed_cross_cov = single_graph.cross_cov(pose1.i, pose2.j)
        return inversed_cross_cov

    def inverse_compound(self, pose1, pose2, robot_idx):
        """Compounding operation for two optimzed Node3D objects. Using calculated
        covariance and cross covariance between the two objects.

        Args:
            Two virtual Edge3D objects, robot_idx being 'a' or 'b'
        Return:
            an Edge3D object
        """
        # Do inversing operation to pose1
        pose1_inversed = self.inverse_op(pose1)
        new_edge = self.compound_op(pose1_inversed, pose2, robot_idx, True)
        return new_edge

    def to_info(self, cov):
        """Convert the covariance matrix to info (21x1 vector in 3D)

        Args:
            A covariance matrix (numpy array)
        Return:
            A vector
        """
        sz = 21                  # size of `info`
        N = 6                  # size of `info_mat`
        info = np.zeros([sz,])
        assert self.is_pos_def(cov)
        info_mat = cholesky_inverse(cov)
        # sym_info_mat = np.maximum(info_mat, info_mat.transpose())
        try:
            assert self.check_symmetry(info_mat)
        except AssertionError:
            # __import__("pdb").set_trace()
            print("info matrix becomes asymmetric")
            # pass
        start = 0
        for i in range(N):
            info[start: start + N - i] = info_mat[i, i:]
            start += N - i
        return info

    def test_inverse_op(self):
        """A simple test case for inverse_op function
        """
        t = [1, 0, 0]
        q = [0, 0, 0, 1]
        cov = np.identity(6)
        cov[3, 3] = 1e-10
        cov[4, 4] = 1e-10
        cov[5, 5] = 1e-10
        # info_mat = np.linalg.inv(cov)
        info = self.to_info(cov)
        edge = Edge3D(1, 2, np.asarray(t), np.asarray(q), info)
        inv_edge = self.inverse_op(edge)
        print("The original edge's covariance matrix: \n")
        print(edge.cov())
        print("The original edge's info matrix:\n")
        print(edge.info_mat())
        print("The inversed edge's covariance matrix: \n")
        print(inv_edge.cov())
        print("The inversed edge's info matrix: \n")
        print(inv_edge.info_mat())
        __import__("pdb").set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the adjacency matrix given one g2o file")
    parser.add_argument("input_fpath", metavar="city10000.g2o", type=str, nargs='?',
                        default="../process_g2o/output.g2o", help="g2o file path")
    parser.add_argument("output_fpath", metavar="adjacency.mtx", type=str, nargs='?',
                        default="adjacency.mtx", help="adjacency file path")
    parser.add_argument("dim", metavar="int", type=int, nargs='?', default=2,
                        help="Using 2D or 3D pose graph")
    args = parser.parse_args()
    if args.dim == 3:
        graph = MultiRobotGraph3D()
    elif args.dim == 2:
        graph = MultiRobotGraph2D()

    graph.read_from(args.input_fpath)
    graph.print_summary()
    print("========== Multi Robot g2o Graph Summary ================")
    graph.print_summary()
    if args.dim == 3:
        ADJ = AdjacencyMatrix3D(graph, gamma=0.1, optim=True)
    elif args.dim == 2:
        ADJ = AdjacencyMatrix(graph, gamma=0.1, optim=True)
    # adj.single_graphs_optimization()
    # ADJ.test_inverse_op()
    coo_adj_mat = ADJ.build_adjacency_matrix()
    io.mmwrite(args.output_fpath, coo_adj_mat, field='integer', symmetry='symmetric')
