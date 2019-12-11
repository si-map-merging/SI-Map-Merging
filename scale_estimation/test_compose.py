from __future__ import print_function

import math
import numpy as np
import gtsam
import matplotlib.pyplot as plt
import gtsam.utils.plot as gtsam_plot

from SIFactor import *

xij = gtsam.Pose2(1.,0.,0.2)
xjk = gtsam.Pose2(2.,2.,0.4)

noiseij = gtsam.noiseModel_Diagonal.Sigmas(np.array([1., 1., 0.5],dtype = np.float)).covariance()
noisejk = gtsam.noiseModel_Diagonal.Sigmas(np.array([1., 1., 0.5],dtype = np.float)).covariance()

print(compound_pose(xij,xjk,noiseij,noisejk))