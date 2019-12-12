from __future__ import print_function

import math

import numpy as np

import gtsam

import matplotlib.pyplot as plt
import gtsam.utils.plot as gtsam_plot

T11 = gtsam.Pose2(0, 0, 0.2)
T12 = gtsam.Pose2(1, 0, 0.1)

# H1=np.zeros((3,3))
# H2=np.zeros((3,3))

def cov_delta_xij(xi,xj,cov):
    _,H1,H2 = between(xi,xj)
    A = np.hstack([H1,H2])
    return A @ cov @ A.T

def cov_delta_xijk(xi,xj,cov):
    pass

def between(p1,p2):
    result = p1.inverse().compose(p2)
    H1 = -result.inverse().AdjointMap()
    size = H1.shape[0]
    H2 = np.eye(size)
    return result,H1,H2


_,H1,H2 = between(T11,T12)
print(np.hstack([H1,H2]))

print(T11.between(T12))
# print(H1)
# print(H2)