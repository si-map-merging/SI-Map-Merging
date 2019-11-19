# Process city10000.g2o to split into two robot trajectories
import random
import math


class Vertex:
    """ A robot pose
    """
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta


class Edge:
    """ A measurement between two vertices
    """
    def __init__(self, i, j, x, y, theta, info):
        self.i = i
        self.j = j
        self.x = x
        self.y = y
        self.theta = theta
        self.info = info


class OdomEdge(Edge):
    """ A odometry measurement (between i and i+1)
    """
    pass


class LoopClosureEdge(Edge):
    """ A loop closure measurement (between i and arbitrary j)
    """
    pass


# map of vertices {id: vertex}
vertices = {}
# list of odometry edges
odom_edges = []
# list of loop closure edges
loop_closure_edges = []


def process_line(line):
    """ Process a single line of file, read in as vertex or edge
    :param line: string
    """
    values = line.split()
    tag = values[0]
    if tag == "VERTEX_SE2":
        id_, x, y, theta = values[1:]
        vertices[int(id_)] = Vertex(float(x), float(y), float(theta))
    elif tag == "EDGE_SE2":
        i, j = [int(x) for x in values[1:3]]
        x, y, theta = [float(x) for x in values[3:6]]
        info = [float(x) for x in values[6:]]
        if i == j-1:
            odom_edges.append(OdomEdge(i, j, x, y, theta, info))
        else:
            loop_closure_edges.append(LoopClosureEdge(i, j, x, y, theta,
                                                      info))
    else:
        raise Exception("Line with unknown tag")


def summarize():
    """ Print summary of original g2o file
    """
    print("===== Original data ========")
    print("# Vertices: {}".format(len(vertices)))
    print("# Odometry Edges: {}".format(len(odom_edges)))
    print("# Loop Closure Edges: {}".format(len(loop_closure_edges)))


filepath = "./dataset/city10000.g2o"
with open(filepath) as fp:
    line = fp.readline()
    while line:
        process_line(line)
        line = fp.readline()

summarize()

# Split trajectory into two, one for each robot
robot_a_vertices = {}
robot_b_vertices = {}
for k, v in vertices.items():
    if k < len(vertices)//2:
        robot_a_vertices[k] = v
    else:
        robot_b_vertices[k] = v

# Split odometry into two, one for for each robot
robot_a_odom = []
robot_b_odom = []
for odom in odom_edges:
    if odom.i in robot_a_vertices and odom.j in robot_a_vertices:
        robot_a_odom.append(odom)
    elif odom.i in robot_b_vertices and odom.j in robot_b_vertices:
        robot_b_odom.append(odom)

# Split loop closures into two, one for each robot
robot_a_lc = []
robot_b_lc = []
inter_lc = []
for lc in loop_closure_edges:
    if lc.i in robot_a_vertices and lc.j in robot_a_vertices:
        robot_a_lc.append(lc)
    elif lc.i in robot_b_vertices and lc.j in robot_b_vertices:
        robot_b_lc.append(lc)
    else:
        inter_lc.append(lc)

# Randomly select inter-robot loop closures
# Select 15, as in paper
N_inliers = 15
assert(len(inter_lc) >= N_inliers)
inter_lc = random.sample(inter_lc, N_inliers)

# Randomly add 90 outliers
N_outliers = 90


def generate_random(n=1):
    # x value
    x_mu = random.uniform(-5, 5)
    x_sigma = random.uniform(-2, 2)

    # y value
    y_mu = random.uniform(-5, 5)
    y_sigma = random.uniform(-2, 2)

    # theta value
    theta_mu = random.uniform(-math.pi, math.pi)
    theta_sigma = random.uniform(-0.5, 0.5)

    # info matrix
    info = [1/x_sigma, 0, 0, 1/y_sigma, 0, 1/theta_sigma]

    # Generate results
    if n == 1:
        return LoopClosureEdge(random.choice(list(robot_a_vertices)),
                               random.choice(list(robot_b_vertices)),
                               random.normalvariate(x_mu, x_sigma),
                               random.normalvariate(y_mu, y_sigma),
                               random.normalvariate(theta_mu, theta_sigma),
                               info)
    else:
        return [LoopClosureEdge(random.choice(list(robot_a_vertices)),
                                random.choice(list(robot_b_vertices)),
                                random.normalvariate(x_mu, x_sigma),
                                random.normalvariate(y_mu, y_sigma),
                                random.normalvariate(theta_mu, theta_sigma),
                                info)
                for _ in range(n)]


# List of loop closure measurements (with noisy data)
inter_lc += [generate_random() for _ in range(N_outliers)]

# Generate perceptual aliasing data
N_aliasing = 5
M = 2  # number of groups of perceptual aliasing data
for _ in range(M):
    inter_lc += generate_random(N_aliasing)

print("===== Generated data ========")
print("# Inter-robot loop closures: {}".format(len(inter_lc)))
assert(len(inter_lc) == N_inliers + N_outliers + M*N_aliasing)
