import numpy as np
from .SIFactor import *

class ScaleEstimation(object):
    """docstring for ScaleEstimation"""
    def __init__(self, lc_num):

        self.lc_num = lc_num
        self.history = []

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





