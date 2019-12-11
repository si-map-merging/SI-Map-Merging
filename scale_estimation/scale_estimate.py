import numpy as np
from .SIFactor import *

class ScaleEstimation(object):
    """docstring for ScaleEstimation"""
    def __init__(self, lc_num):

        self.lc_num = lc_num
        self.history = []

    def get_reletive_pose(self, i , j):

        pi = self.result.atPose2(i)
        pj = self.result.atPose2(j)
        key_vec = gtsam.gtsam.KeyVector()
        key_vec.push_back(i)
        key_vec.push_back(j)
        cov = self.marginals.jointMarginalCovariance(key_vec).fullMatrix()
        noise = cov_delta_xij(pi,pj,cov)
        return pi.between(pj), noise


    def scale_estimate(self, poses, covs, index_list):

        xa_list, xb_list, z_list = poses
        qa_list, qb_list, qab_list = covs

        graph = gtsam.NonlinearFactorGraph()

        PRIOR_NOISE = gtsam.noiseModel_Diagonal.Sigmas(np.array([0.1, 0.1, 0.1],dtype = np.float))
        graph.add(gtsam.PriorFactorPose2(1, gtsam.Pose2(0.0, 0.0, 0.0), PRIOR_NOISE))

        graph.add(gtsam.BetweenFactorPose2(1, 2, xa_list[0], qa_list[0]))
        graph.add(gtsam.BetweenFactorPose2(2, 3, xa_list[1], qa_list[1]))
        graph.add(gtsam.BetweenFactorPose2(1, 3, xa_list[2], qa_list[2]))

        measured,noise,H=construct_SIFactor2(xb_list[0],qb_list[0])
        graph.add(gtsam.SIBetweenFactorPose2(4, 5, measured, noise, H))

        measured,noise,H=construct_SIFactor2(xb_list[1],qb_list[1])
        graph.add(gtsam.SIBetweenFactorPose2(5, 6, measured, noise, H))

        measured,noise,H=construct_SIFactor2(xb_list[2],qb_list[2])
        graph.add(gtsam.SIBetweenFactorPose2(4, 6, measured, noise, H))

        measured,noise,H=construct_SIFactor2(z_list[0],qab_list[0])
        graph.add(gtsam.SIBetweenFactorPose2(1, 4, measured, noise, H))

        measured,noise,H=construct_SIFactor2(z_list[1],qab_list[1])
        graph.add(gtsam.SIBetweenFactorPose2(2, 5, measured, noise, H))

        measured,noise,H=construct_SIFactor2(z_list[2],qab_list[2])
        graph.add(gtsam.SIBetweenFactorPose2(3, 6, measured, noise, H))

        initial_estimate = gtsam.Values()
        initial_estimate.insert(1, gtsam.Pose2(0.5, 0.0, 0.3))
        initial_estimate.insert(2, gtsam.Pose2(2.3, 0.1, -0.6))
        initial_estimate.insert(3, gtsam.Pose2(2.3, 2.1, -0.5))
        initial_estimate.insert(4, gtsam.Pose2(0.2, 3.0, 0.2))
        initial_estimate.insert(5, gtsam.Pose2(5.8, 2.1, -0.6))
        initial_estimate.insert(6, gtsam.Pose2(6.3, 1.1, -0.3))

        params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
        result = optimizer.optimize()

        self.result = result
        self.marginals = gtsam.Marginals(graph, result)

        x45,noise45 = self.get_reletive_pose(4,5)
        x56,noise56 = self.get_reletive_pose(5,6)
        x46,noise46 = self.get_reletive_pose(4,6)

        z14,noise14 = self.get_reletive_pose(1,4)
        z25,noise25 = self.get_reletive_pose(2,5)
        z36,noise36 = self.get_reletive_pose(3,6)


        qb_list = [noise45,noise56,noise46]
        new_xb_list = [x45,x56,x46]


        sb,sb_std = get_scale3(new_xb_list, xb_list, qb_list)
        s1,s1_std = get_scale(z14, z_list[0], noise14)
        s2,s2_std = get_scale(z25, z_list[1], noise25)
        s3,s3_std = get_scale(z36, z_list[2], noise36)

        s_list = [sb,s1,s2,s3]
        std_list = [sb_std, s1_std, s2_std, s3_std]
        return s_list,std_list

        # self.history





