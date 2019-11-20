import math
import numpy as np
import convert_to_multi
import pdb

def inverse_op(pose, cov):
    """
    Compute x_ji given x_ij
    Input: A 3x1 pose vector [x, y, phi]
           The covariance matrix Cov of that pose, cov is a np array
    """
    x, y, phi = pose
    new_pose = [-x*math.cos(phi)-y*math.sin(phi),
                x*math.sin(phi)-y*math.cos(phi),
                -phi]
    J_minus = np.matrix([[-math.cos(phi) -math.sin(phi) y], \
                         [math.sin(phi) -math.cos(phi) -x], \
                         [0              0             -1]])
    new_cov = np.matmul(np.matmul(J_, cov), np.transpose(J_))
    return new_pose, new_cov

def compound_op(pose1, pose2, cov1, cov2, cov_cross):
    """
    Compute pose1 circle+ pose2
    Input: pose1 = [x1, y1, phi1]
           pose2 = [x2, y2, phi2]
    """
    x1, y1, phi1 = pose1
    x2, y2, phi2 = pose2
    new_pose = [x2*math.cos(phi1) - y2*math.sin(phi1) + x1,
                x2*math.sin(phi1) + y2*math.cos(phi1) + y1,
                phi1 + phi2]
    prev_cov = np.matrix([[cov1                 cov_cross],
                          [np.transpose(cov_cross)   cov2]])
    J_plus = np.matrix([[1 0 -(new_pose[1]-y1) math.cos(phi1) -math.sin(phi1) 0], \
                        [0 1 (new_pose[0]-x1) math.sin(phi1) -math.cos(phi1)  0], \
                        [0 0       1              0               0           1]])
    new_cov = np.matmul(np.matmul(J_plus, prev_cov), np.transpose(J_plus))
    return new_pose, new_cov

def compute_mahalanobis_distance():
    pass

class CrossPose(Edge):
    pass

class AdjacencyMatrix:
    def __init__(self, graphA, graphB, gamma=1):
        self.gamma = gamma
        self.graphA = graphA
        self.graphB = graphB
        assert len(graphA.nodes) = len(graphB.nodes)
        self.N = len(graphA.nodes)
        # self.adjacency_matrix = np.zeros((self.num_measurements, self.num_measurements))

    def build_cross_traj_poses_vec(self):
        cross_pose_vector = np.zeros(self.N * self.N)
        nNodes = 0
        for idx1, pose1 in graphA.nodes:
            for idx2, pose2 in graphB.nodes:
                cross_pose = self.compute_cross_pose(pose1, pose2)
                cross_pose_vector[nNodes] = cross_pose
                nNodes = nNodes + 1
        return cross_pose_vector

    def compute_cross_pose(self, pose1, pose2):
        """
        return a CrossPose object, which is inheriting from Edge
        """
        pass

    def build_adjacency_matrix(self):
        adjacency_matrix = np.zeros((self.N * self.N, self.N * self.N))
        for i in range(self.N * self.N):
            for j in range(self.N * self.N):
                mahl = self.compute_mahalanobis_distance(cross_pose_vector[i], 
                                                         cross_pose_vector[j])
                if mahl <= self.gamma:
                    adjacency_matrix[i, j] = 1
        return adjacency_matrix

    def compute_mahalanobis_distance(self, pose1, pose2):
        return 1
    
    def get_covariance():
        return np.identity(3)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the adjacency matrix given two g2o files")
    parser.add_argument("input_fpath1", metavar="input1.g2o", type=str,
                        help="first g2o file path")
    parser.add_argument("input_fpath2", metavar="input2.g2o", type=str,
                        help="second g2o file path")
    parser.add_argument("output_fpath", metavar="adjacency.txt", type=str,
                        default="adjacency.txt", help="adjacency file path")
    args = parser.parse_args()

    graphA = SingleRobotGraph(args.input_fpath1)
    graphB = SingleRobotGraph(args.input_fpath2)
    adj = AdjacencyMatrix(graphA, graphB, 1)
    adjMatrix = adj.build_adjacency_matrix()