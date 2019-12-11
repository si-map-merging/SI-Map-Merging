from __future__ import print_function

import math
import numpy as np
import gtsam
import matplotlib.pyplot as plt
import gtsam.utils.plot as gtsam_plot


def cov_delta_xij(xi, xj, joint_marginal_matrix):
    _,H1,H2 = between(xi,xj)
    A = np.hstack([H1,H2])
    return A @ joint_marginal_matrix @ A.T


def between(p1, p2):
    result = p1.inverse().compose(p2)
    H1 = -result.inverse().AdjointMap()
    size = H1.shape[0]
    H2 = np.eye(size)
    return result,H1,H2

def get_scale(x,x_original,cov):
    
    return s,var
    pass

def inv_Q(Q):
    NOISE = gtsam.noiseModel_Gaussian.Covariance(Q)
    Q_inv = NOISE.information()
    return Q_inv

def construct_SIFactor2(measured,cov):
    #measured = xi.between(xj)
    # print(type(joint_marginal_matrix))

    #cov = cov_delta_xij(xi,xj,joint_marginal_matrix)
    if not (type(cov) is type(np.array([1.,2.]))):
        cov = cov.covariance()

    size = cov.shape[0]
    dim = measured.translation().vector().shape[0]
    # print(ti.shape[0])


    R_ij = measured.rotation().matrix()
    Jt = np.hstack([R_ij,np.zeros((dim,size-dim))])

    #print(Jt)

    J = np.block([[Jt],
                [np.zeros((size-dim,dim)),np.eye(size-dim)]])
    #print(J)

    Q = Jt @ cov @ Jt.T
    Q_inv = inv_Q(Q)

    #print(Q_inv)

    xij = measured.translation().vector().reshape((dim,-1))
    #print(xij.shape)

    Ht = np.eye(dim) - (xij @ xij.T @ Q_inv)/(xij.T @ Q_inv @ xij)
    #print(Ht)
    #print(np.linalg.eig(Ht))

    #H is not full rank, reduce dimesion by using the eigenvector with eigenvalue 1
    eigenValues, eigenVectors = np.linalg.eig(Ht)
    #print('eigen')
    #print(eigenVectors,eigenValues)
    idx = eigenValues.argsort()[::-1]
    #print(idx)
    #eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx[:-1]]
    #print(eigenVectors)
    #Ht = eigenVectors.T
    Ht = Ht[0:dim-1,:]#*0.01
    #print(Ht)

    H = np.block([
        [Ht, np.zeros((Ht.shape[0],size-dim))],
        [np.zeros((size-dim,Ht.shape[1])),np.eye(size-dim)]])

    #print(H)
    print(measured)

    H = H @ J
    print(H)

    noise = H @ J @ cov @ J.T @ H.T
    #noise = J@cov@J.T
    print(noise)

    print()
    return measured,gtsam.noiseModel_Gaussian.Covariance(noise),H


    # print(np.linalg.matrix_rank(H))

def scale_normalizae(p, Q, t=1.):
    translation = p.translation().vector()
    s = 1./translation.norm()
    p = scale_pose(p,s)
    Q = scale_covariance(Q,s)
    return p,Q
    pass

def scale_pose(p,s):
    #print(p)
    T_trans = type(p.translation())
    T_pose = type(p)
    #print(p.translation())
    translation = p.translation().vector()
    rotation = p.rotation()

    translation = T_trans(translation*s)
    #translation = T(translation)
    new_p = T_pose(r=rotation, t=translation)
    #print(new_p)
    return new_p

def scale_covariance(cov, s):
    #print(type(cov))
    if not (type(cov) is type(np.array([1.,2.]))):
        cov = cov.covariance()
    #print(cov)
    size = cov.shape[0]
    if size == 6:
        dim = 3
    elif size == 3:
        dim = 2
    else:
        print('unkown covariance size')
    # print(type(covariance))
    scale_matrix = np.block([
        [np.eye(dim)*s,np.zeros((dim,size-dim))],
        [np.zeros((size-dim,dim)),np.eye(size-dim)]
        ])
    #print(scale_matrix)

    scaled_cov = scale_matrix @ cov @ scale_matrix
    #print(scaled_cov)
    return gtsam.noiseModel_Gaussian.Covariance(scaled_cov)

    #print(translation)
    #return(measured,noise,H)

if __name__ == '__main__':
    graph = gtsam.NonlinearFactorGraph()

    sb = np.zeros((3,3),dtype = np.float)
    np.fill_diagonal(sb, [0.2*0.2,0.5*0.5,0.1*0.1])
    #print(sb)
    ODOMETRY_NOISE = gtsam.noiseModel_Gaussian.Covariance(sb)

    PRIOR_NOISE = gtsam.noiseModel_Diagonal.Sigmas(np.array([0.6, 0.2, 0.1],dtype = np.float))
    graph.add(gtsam.PriorFactorPose2(1, gtsam.Pose2(0.0, 0.0, 1.0), PRIOR_NOISE))

    graph.add(gtsam.BetweenFactorPose2(1, 2, gtsam.Pose2(2.0, 0.0, 0.5), ODOMETRY_NOISE))

    graph.add(gtsam.BetweenFactorPose2(2, 3, gtsam.Pose2(0.0, 2.0, 0.2), ODOMETRY_NOISE))

    graph.add(gtsam.BetweenFactorPose2(3, 1, gtsam.Pose2(-2.0, -2.0, -0.3), ODOMETRY_NOISE))

    initial_estimate = gtsam.Values()
    initial_estimate.insert(1, gtsam.Pose2(0.5, 0.0, 0.2))
    initial_estimate.insert(2, gtsam.Pose2(2.3, 0.1, -0.2))
    initial_estimate.insert(3, gtsam.Pose2(2.3, 2.1, -0.2))

    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()

    marginals = gtsam.Marginals(graph, result)
    # for i in range(1, 4):
    #     print("X{} covariance:\n{}\n".format(i, marginals.marginalCovariance(i)))

    fig = plt.figure(0)
    for i in range(1, 4):
        gtsam_plot.plot_pose2(0, result.atPose2(i), 0.5, marginals.marginalCovariance(i))

    plt.axis('equal')


    # test SI

    #PRIOR_NOISE = gtsam.noiseModel_Diagonal.Sigmas(np.array([1.6, 0.6, 0.001],dtype = np.float))

    p1 = result.atPose2(1)
    p2 = result.atPose2(2)
    p3 = result.atPose2(3)



    marginals = gtsam.Marginals(graph, result)

    key_vec = gtsam.gtsam.KeyVector()
    key_vec.push_back(1)
    key_vec.push_back(2)
    cov = marginals.jointMarginalCovariance(key_vec).fullMatrix()
    NOISE12 = cov_delta_xij(p1,p2,cov)

    marginals = gtsam.Marginals(graph, result)
    key_vec = gtsam.gtsam.KeyVector()
    key_vec.push_back(2)
    key_vec.push_back(3)
    cov = marginals.jointMarginalCovariance(key_vec).fullMatrix()
    NOISE23 = cov_delta_xij(p2,p3,cov)

    marginals = gtsam.Marginals(graph, result)
    key_vec = gtsam.gtsam.KeyVector()
    key_vec.push_back(1)
    key_vec.push_back(3)
    cov = marginals.jointMarginalCovariance(key_vec).fullMatrix()
    NOISE13 = cov_delta_xij(p1,p3,cov)


    graph = gtsam.NonlinearFactorGraph()
    graph.add(gtsam.PriorFactorPose2(1, p1, PRIOR_NOISE))
    graph.add(gtsam.PriorFactorPose2(2, p2, PRIOR_NOISE))
    scale = 0.4
    print('scale: ',scale)

    p1 = scale_pose(p1,scale)
    p2 = scale_pose(p2,scale)
    p3 = scale_pose(p3,scale)

    NOISE12 = scale_covariance(NOISE12,scale)
    NOISE23 = scale_covariance(NOISE23,scale)
    NOISE13 = scale_covariance(NOISE13,scale)


    #scale_covariance(PRIOR_NOISE,2)

    # x12 = between(p1,p2)
    # x23 = between(p2,p3)
    # x13 = between(p1,p3)
    
    measured,noise,H=construct_SIFactor2(p1.between(p2),NOISE12)
    graph.add(gtsam.SIBetweenFactorPose2(1, 2, measured, noise, H))

    measured,noise,H=construct_SIFactor2(p2.between(p3),NOISE23)
    graph.add(gtsam.SIBetweenFactorPose2(2, 3, measured, noise, H))

    measured,noise,H=construct_SIFactor2(p1.between(p3),NOISE13)
    graph.add(gtsam.SIBetweenFactorPose2(1, 3, measured, noise, H))


    #x12 = scale_pose(x12,scale)

    #print('test scale')
    # scale_covariance(,2)

    # initial_estimate = gtsam.Values()
    # initial_estimate.insert(1, gtsam.Pose2(0.5, 0.0, 0.2))
    # initial_estimate.insert(2, gtsam.Pose2(2.3, 0.1, -0.2))
    # initial_estimate.insert(3, gtsam.Pose2(5.3, 2.1, -0.2))

    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()

    marginals = gtsam.Marginals(graph, result)
    # for i in range(1, 4):
    #     print("X{} covariance:\n{}\n".format(i, marginals.marginalCovariance(i)))

    fig = plt.figure(0)
    for i in range(1, 4):
        gtsam_plot.plot_pose2(0, result.atPose2(i), 0.5, marginals.marginalCovariance(i))

    plt.axis('equal')

    plt.show()

