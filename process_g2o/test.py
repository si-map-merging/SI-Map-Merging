import numpy as np
import quaternion


def ZYZ(phi, theta, psi):
    RotZ1 = np.array([[np.cos(phi), -np.sin(phi), 0],
                     [np.sin(phi), np.cos(phi), 0],
                     [0, 0, 1]])
    RotY = np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])
    RotZ2 = np.array([[np.cos(psi), -np.sin(psi), 0],
                     [np.sin(psi), np.cos(psi), 0],
                     [0, 0, 1]])

    return np.matmul(np.matmul(RotZ1, RotY), RotZ2)

# q = quaternion.quaternion(0.2, 0.3, 0.1, 0.5)
# print(q)
# print(quaternion.as_rotation_matrix(q))
# angles = quaternion.as_euler_angles(q)
# print(angles)
# print(ZYZ(*angles))

from utils import Pose2D
import random
import math

def random_pose():
    x_mu = random.uniform(-5, 5)
    x_sigma = 0.1
    y_mu = random.uniform(-5, 5)
    y_sigma = 0.1
    theta_mu = random.uniform(-math.pi, math.pi)
    theta_sigma = 0.1

    x = random.gauss(x_mu, x_sigma)
    y = random.gauss(y_mu, y_sigma)
    theta = random.gauss(theta_mu, theta_sigma)

    return Pose2D.from_x_y_theta(x, y, theta)

def relative(pose1, pose2):
    return pose1.inverse() * pose2

a = random_pose()
b = random_pose()
c = random_pose()
d = random_pose()

ab = random_pose()
# cd = relative(c, d)

cd_est = relative(c, a) * ab * relative(b, d)
print(cd_est)

loop_pose = cd_est.inverse() * relative(c, a) * ab * relative(b, d)
print(loop_pose)