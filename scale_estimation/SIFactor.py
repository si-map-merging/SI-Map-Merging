from __future__ import print_function

import math
import numpy as np
import gtsam
import matplotlib.pyplot as plt
import gtsam.utils.plot as gtsam_plot



def cov_delta_xij(xi, xj, joint_marginal_matrix):
    """Get covariance of delta xij
    """
    _,H1,H2 = between(xi,xj)
    A = np.hstack([H1,H2])
    return A @ joint_marginal_matrix @ A.T


def between(p1, p2):
    """Get between factor of pose 1 and pose 2
    """
    result = p1.inverse().compose(p2)
    H1 = -result.inverse().AdjointMap()
    size = H1.shape[0]
    H2 = np.eye(size)
    return result,H1,H2

def get_reletive_pose(result,marginals, i , j):
    """Get relative pose between i and j
    """
    pi = result.atPose2(i)
    pj = result.atPose2(j)
    key_vec = gtsam.gtsam.KeyVector()
    key_vec.push_back(i)
    key_vec.push_back(j)
    cov = marginals.jointMarginalCovariance(key_vec).fullMatrix()
    noise = cov_delta_xij(pi,pj,cov)
    return pi.between(pj), noise

def compound_pose(xij,xjk,noiseij,noisejk):
    """Compound the pose together
    """

    graph = gtsam.NonlinearFactorGraph()

    PRIOR_NOISE = gtsam.noiseModel_Diagonal.Sigmas(np.array([0.0001, 0.0001, 0.0001],dtype = np.float))
    graph.add(gtsam.PriorFactorPose2(1, gtsam.Pose2(0.0, 0.0, 0.0), PRIOR_NOISE))

    noiseij = gtsam.noiseModel_Gaussian.Covariance(noiseij)
    noisejk = gtsam.noiseModel_Gaussian.Covariance(noisejk)

    graph.add(gtsam.BetweenFactorPose2(1, 2, xij, noiseij))
    graph.add(gtsam.BetweenFactorPose2(2, 3, xjk, noisejk))

    initial_estimate = gtsam.Values()
    initial_estimate.insert(1, gtsam.Pose2(0.5, 0.0, 0.2))
    initial_estimate.insert(2, gtsam.Pose2(2.3, 0.1, -0.2))
    initial_estimate.insert(3, gtsam.Pose2(2., 0.1, -0.2))

    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()

    marginals = gtsam.Marginals(graph, result)

    return get_reletive_pose(result, marginals, 1, 3)


def inv_Q(Q):
    """Take inverse on positive-definite matrix
    """
    NOISE = gtsam.noiseModel_Gaussian.Covariance(Q)
    Q_inv = NOISE.information()
    return Q_inv

def rotation_matrix_between_vectors(a,b):
    dim=3
    a = a.reshape(-1)
    b = b.reshape(-1)
    if a.shape[0] == 2:
        dim = 2
        a = np.hstack([a,[0]])
    if b.shape[0] == 2:
        b = np.hstack([b,[0]])
    a = a/np.linalg.norm(a)
    b = b/np.linalg.norm(b)

    v = np.cross(a,b)
    s = np.linalg.norm(v)
    c = a.dot(b)
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0.],
        ])
    if s==0:
        R = np.eye(3)
    else:
        R = np.eye(3) + vx + vx @ vx *(1-c)/s**2
    if dim ==2:
        R = R[0:2,0:2]
    return R

def marginalization(M, index, inverse = inv_Q):
    shape = M.shape[0]
    ind1 = index
    ind2 = []
    for i in range(shape):
        if i not in index:ind2.append(i)
    A = M[ind1,:][:,ind1]
    B = M[ind1,:][:,ind2]
    BT = M[ind2,:][:,ind1]
    C = M[ind2,:][:,ind2]
    result = A-B@inverse(C)@BT
    return result

def get_scale3(x_list,x_original_list,cov_list):
    s_list = []
    std_list = []
    for i in range(len(x_list)):
        s,std_s = get_scale(x_list[i],x_original_list[i],cov_list[i])
        s_list.append(s)
        std_list.append(std_s)
    s = np.array(s_list)
    std = np.array(std_list)

    w = np.sum(1./(std**2))
    s_mean = np.sum(s/(std**2))/w
    std = 1./np.sqrt(w)
    return s_mean,std


def get_scale(x,x_original,cov):
    size = cov.shape[0]
    dim = x.translation().vector().shape[0]

    cov_translation = inv_Q(marginalization(inv_Q(cov), range(dim)))
    R_ij = x.rotation().matrix()

    ex = np.zeros(dim)
    ex[0] = 1
    R_ki = rotation_matrix_between_vectors(x.translation().vector(), ex)

    R_kj = R_ki @ R_ij


    new_cov = R_kj @ cov_translation

    var = inv_Q(marginalization(inv_Q(new_cov),[0]))
    var = var.reshape(-1)[0]

    l = np.linalg.norm(x.translation().vector())
    l0 = np.linalg.norm(x_original.translation().vector())

    std_l = np.sqrt(var)
    s = l0/l
    std_s = s/l *std_l

    return s,std_s

def get_length(x,x_original,cov):
    size = cov.shape[0]
    dim = x.translation().vector().shape[0]

    cov_translation = inv_Q(marginalization(inv_Q(cov), range(dim)))

    R_ij = x.rotation().matrix()

    ex = np.zeros(dim)
    ex[0] = 1
    R_ki = rotation_matrix_between_vectors(x.translation().vector(), ex)

    R_kj = R_ki @ R_ij

    new_cov = R_kj @ cov_translation

    var = inv_Q(marginalization(inv_Q(new_cov),[0]))
    var = var.reshape(-1)[0]

    l = np.linalg.norm(x.translation().vector())
    l0 = np.linalg.norm(x_original.translation().vector())

    std_l = np.sqrt(var)
    return l,std_l


def construct_SIFactor2(measured,cov):
    """Construct SIBetweenFactor for Pose2
    """
    if not (type(cov) is type(np.array([1.,2.]))):
        cov = cov.covariance()

    size = cov.shape[0]
    dim = measured.translation().vector().shape[0]

    R_ij = measured.rotation().matrix()
    Jt = np.hstack([R_ij,np.zeros((dim,size-dim))])


    J = np.block([[Jt],
                [np.zeros((size-dim,dim)),np.eye(size-dim)]])
    Q = Jt @ cov @ Jt.T
    Q_inv = inv_Q(Q)


    xij = measured.translation().vector().reshape((dim,-1))

    Ht = np.eye(dim) - (xij @ xij.T @ Q_inv)/(xij.T @ Q_inv @ xij)

    eigenValues, eigenVectors = np.linalg.eig(Ht)
    idx = eigenValues.argsort()[::-1]
    eigenVectors = eigenVectors[:,idx[:-1]]
    Ht = Ht[0:dim-1,:]

    H = np.block([
        [Ht, np.zeros((Ht.shape[0],size-dim))],
        [np.zeros((size-dim,Ht.shape[1])),np.eye(size-dim)]])

    H = H @ J

    noise = H @ J @ cov @ J.T @ H.T
    return measured,gtsam.noiseModel_Gaussian.Covariance(noise),H


def scale_normalizae(p, Q, t=1.):
    translation = p.translation().vector()
    s = 1./translation.norm()
    p = scale_pose(p,s)
    Q = scale_covariance(Q,s)
    return p,Q
    pass

def scale_pose(p,s):
    T_trans = type(p.translation())
    T_pose = type(p)
    translation = p.translation().vector()
    rotation = p.rotation()

    translation = T_trans(translation*s)
    new_p = T_pose(r=rotation, t=translation)
    return new_p

def scale_covariance(cov, s):
    if not (type(cov) is type(np.array([1.,2.]))):
        cov = cov.covariance()
    size = cov.shape[0]
    if size == 6:
        dim = 3
    elif size == 3:
        dim = 2
    else:
        print('unkown covariance size')
    scale_matrix = np.block([
        [np.eye(dim)*s,np.zeros((dim,size-dim))],
        [np.zeros((size-dim,dim)),np.eye(size-dim)]
        ])

    scaled_cov = scale_matrix @ cov @ scale_matrix
    return gtsam.noiseModel_Gaussian.Covariance(scaled_cov)


if __name__ == '__main__':
    graph = gtsam.NonlinearFactorGraph()

    sb = np.zeros((3,3),dtype = np.float)
    np.fill_diagonal(sb, [0.2*0.2,0.5*0.5,0.1*0.1])
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
    fig = plt.figure(0)
    for i in range(1, 4):
        gtsam_plot.plot_pose2(0, result.atPose2(i), 0.5, marginals.marginalCovariance(i))

    plt.axis('equal')


    p1 = result.atPose2(1)
    p2 = result.atPose2(2)
    p3 = result.atPose2(3)



    marginals = gtsam.Marginals(graph, result)

    key_vec = gtsam.gtsam.KeyVector()
    key_vec.push_back(1)
    key_vec.push_back(2)
    cov = marginals.jointMarginalCovariance(key_vec).fullMatrix()
    NOISE12 = cov_delta_xij(p1,p2,cov)

    key_vec = gtsam.gtsam.KeyVector()
    key_vec.push_back(2)
    key_vec.push_back(3)
    cov = marginals.jointMarginalCovariance(key_vec).fullMatrix()
    NOISE23 = cov_delta_xij(p2,p3,cov)

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

    x_original_list = [p1.between(p2), p2.between(p3), p1.between(p3)]

    NOISE12 = scale_covariance(NOISE12,scale)
    NOISE23 = scale_covariance(NOISE23,scale)
    NOISE13 = scale_covariance(NOISE13,scale)

    measured,noise,H=construct_SIFactor2(p1.between(p2),NOISE12)
    graph.add(gtsam.SIBetweenFactorPose2(1, 2, measured, noise, H))

    measured,noise,H=construct_SIFactor2(p2.between(p3),NOISE23)
    graph.add(gtsam.SIBetweenFactorPose2(2, 3, measured, noise, H))

    measured,noise,H=construct_SIFactor2(p1.between(p3),NOISE13)
    graph.add(gtsam.SIBetweenFactorPose2(1, 3, measured, noise, H))

    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()

    marginals = gtsam.Marginals(graph, result)



    p1 = result.atPose2(1)
    p2 = result.atPose2(2)
    p3 = result.atPose2(3)


    marginals = gtsam.Marginals(graph, result)

    key_vec = gtsam.gtsam.KeyVector()
    key_vec.push_back(1)
    key_vec.push_back(2)
    cov = marginals.jointMarginalCovariance(key_vec).fullMatrix()
    NOISE12 = cov_delta_xij(p1,p2,cov)

    key_vec = gtsam.gtsam.KeyVector()
    key_vec.push_back(2)
    key_vec.push_back(3)
    cov = marginals.jointMarginalCovariance(key_vec).fullMatrix()
    NOISE23 = cov_delta_xij(p2,p3,cov)

    key_vec = gtsam.gtsam.KeyVector()
    key_vec.push_back(1)
    key_vec.push_back(3)
    cov = marginals.jointMarginalCovariance(key_vec).fullMatrix()
    NOISE13 = cov_delta_xij(p1,p3,cov)

    cov_list = [NOISE12, NOISE23, NOISE13]

    x12 = p1.between(p2)

    x_list = [p1.between(p2), p2.between(p3), p1.between(p3)]
    print('get scale')
    print(get_scale(x_list[0],x_original_list[0],cov_list[0]))

    print('get scale 3')
    print(get_scale3(x_list,x_original_list,cov_list))

    fig = plt.figure(0)
    for i in range(1, 4):
        gtsam_plot.plot_pose2(0, result.atPose2(i), 0.5, marginals.marginalCovariance(i))

    plt.axis('equal')

    plt.show()

