from __future__ import print_function

import math

import numpy as np

import gtsam

import matplotlib.pyplot as plt
import gtsam.utils.plot as gtsam_plot

def cov_delta_xij(xi,xj,cov):
    _,H1,H2 = between(xi,xj)
    A = np.hstack([H1,H2])
    return A @ cov @ A.T


def between(p1,p2):
    result = p1.inverse().compose(p2)
    H1 = -result.inverse().AdjointMap()
    size = H1.shape[0]
    H2 = np.eye(size)
    return result,H1,H2

graph = gtsam.NonlinearFactorGraph()

sb = np.zeros((3,3),dtype = np.float)
np.fill_diagonal(sb, [0.2*0.2,0.2*0.2,0.1*0.1])
print(sb)
ODOMETRY_NOISE = gtsam.noiseModel_Gaussian.Covariance(sb)
print(ODOMETRY_NOISE.R())
print(ODOMETRY_NOISE.information())

PRIOR_NOISE = gtsam.noiseModel_Diagonal.Sigmas(np.array([0.3, 0.3, 0.1],dtype = np.float))
graph.add(gtsam.PriorFactorPose2(1, gtsam.Pose2(0.0, 0.0, 0.0), PRIOR_NOISE))

graph.add(gtsam.SIBetweenFactorPose2(1, 2, gtsam.Pose2(1.0, 0.0, 1.0), ODOMETRY_NOISE))
graph.add(gtsam.SIBetweenFactorPose2(1, 2, gtsam.Pose2(1.0, 1.0, 0.2), ODOMETRY_NOISE))
graph.add(gtsam.SIBetweenFactorPose2(2, 3, gtsam.Pose2(-1.0, 1.0, 0.0), ODOMETRY_NOISE))
graph.add(gtsam.SIBetweenFactorPose2(3, 1, gtsam.Pose2(0.0, -1.0, -0.5), ODOMETRY_NOISE))

graph.add(gtsam.BetweenFactorPose2(1, 2, gtsam.Pose2(2.0, 0.0, 1.0), ODOMETRY_NOISE))

initial_estimate = gtsam.Values()
initial_estimate.insert(1, gtsam.Pose2(0.5, 0.0, 0.2))
initial_estimate.insert(2, gtsam.Pose2(2.3, 0.1, -0.2))
initial_estimate.insert(3, gtsam.Pose2(4.1, 0.1, math.pi / 2))

print("\nInitial Estimate:\n{}".format(initial_estimate))

params = gtsam.LevenbergMarquardtParams()
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
result = optimizer.optimize()

print("Final Result:\n{}".format(result))

marginals = gtsam.Marginals(graph, result)
key_vec = gtsam.gtsam.KeyVector()
key_vec.push_back(1)
key_vec.push_back(2)
print('joint marginals')
cov = marginals.jointMarginalCovariance(key_vec).fullMatrix()

print('eigen')
w,_ = np.linalg.eig(cov)
print(w)

p1 = result.atPose2(1)
p2 = result.atPose2(2)

cov = cov_delta_xij(p1,p2,cov)
# print(ODOMETRY_NOISE)

# print('eigen')
# w,_ = np.linalg.eig(cov)
# print(w)

print('eigen')
w,_ = np.linalg.eig(cov)
print(w)

marginals = gtsam.Marginals(graph, result)
for i in range(1, 4):
    print("X{} covariance:\n{}\n".format(i, marginals.marginalCovariance(i)))

fig = plt.figure(0)
for i in range(1, 4):
    gtsam_plot.plot_pose2(0, result.atPose2(i), 0.5, marginals.marginalCovariance(i))

plt.axis('equal')
plt.show()